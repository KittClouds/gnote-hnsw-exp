import { HNSW } from './hnsw';
import { tables, events } from '../../livestore/schema';
import { blobToVec, vecToBlob } from './binaryUtils';
import { embeddingService } from './EmbeddingService';
import { Block } from '@blocknote/core';
import { toast } from 'sonner';

interface Centroid {
  id: number;
  vector: Float32Array;
}

interface SearchResult {
  noteId: string;
  title: string;
  content: string;
  score: number;
}

// Helper to convert BlockNote content to plain text
function blocksToText(blocks: Block[] | undefined): string {
  if (!blocks) return '';
  
  return blocks.map(block => {
    if (!block.content) return '';
    
    if (Array.isArray(block.content)) {
      return block.content.map(inline => {
        if (typeof inline === 'object' && inline !== null && 'type' in inline && inline.type === 'text') {
          return (inline as any).text || '';
        }
        return '';
      }).join('');
    }
    
    return '';
  }).join('\n');
}

// L2 normalize a vector to unit length for vector hygiene
function l2Normalize(v: Float32Array): Float32Array {
  let norm = 0; 
  for (const x of v) norm += x * x;
  norm = 1 / Math.sqrt(norm || 1e-9);
  return v.map(x => x * norm) as Float32Array;
}

/**
 * Implements a SPANN-like hybrid search.
 * - An in-memory HNSW index holds a small number of centroids.
 * - The full vector dataset remains on "disk" in the SQLite database.
 * - Search is a two-step process: find candidate clusters in memory, then
 *   fetch and search vectors from only those clusters.
 */
class SpannSearchService {
  private storeRef: any = null;
  private isReady = false;
  private isInitialized = false;

  // In-memory components of the index
  private centroids: Centroid[] = [];
  private centroidIndex: HNSW | null = null;

  constructor(private config = {
    // How many clusters/centroids to partition the dataset into.
    // Reduced from 100 to 5 for easier testing
    numClusters: 5,
    // How many of the closest clusters to search during a query.
    // Higher is more accurate but slower.
    searchProbeCount: 3,
  }) {}

  /**
   * Injects the LiveStore instance. Called once on app startup.
   */
  public setStore(store: any) {
    this.storeRef = store;
    if (!this.isInitialized && store) {
      this.initialize();
      this.isInitialized = true;
    }
  }

  private async initialize() {
    if (this.isReady) return;
    try {
      await embeddingService.ready();
      await this.loadCentroidsFromDB();
      this.isReady = true;
      console.log('SpannSearchService initialized');
      if (!this.centroidIndex) {
        console.warn('SPANN index is not built. Use "Sync & Rebuild Index" to create the index.');
      }
    } catch (error) {
      console.error('Failed to initialize SpannSearchService:', error);
    }
  }

  /**
   * Loads ONLY the centroids from the database into the in-memory HNSW index.
   */
  private async loadCentroidsFromDB() {
    if (!this.storeRef) return;

    try {
      const clusterRows = this.storeRef.query(tables.embeddingClusters.select());
      if (!Array.isArray(clusterRows) || clusterRows.length === 0) {
        this.centroids = [];
        this.centroidIndex = null;
        return;
      }

      this.centroids = clusterRows.map((row: any) => ({
        id: row.id,
        vector: blobToVec(row.vecData, row.vecDim),
      }));

      // Build the fast, in-memory index on the centroids
      this.centroidIndex = new HNSW(16, 200, null, 'cosine');
      const hnswData = this.centroids.map(c => ({ id: c.id, vector: c.vector }));
      await this.centroidIndex.buildIndex(hnswData);

      console.log(`SpannSearchService: Loaded ${this.centroids.length} centroids into memory`);
    } catch (error) {
      console.error('Failed to load centroids from database:', error);
    }
  }

  /**
   * Force cleanup of stale embeddings - standalone method for debugging
   */
  public async forceCleanupStaleEmbeddings() {
    if (!this.storeRef) {
      console.error("SPANN: Store reference not available for cleanup");
      return { removed: 0, errors: [] };
    }

    console.log("SPANN: Starting force cleanup of stale embeddings");

    // Get current notes (source of truth)
    const allNotesResult = this.storeRef.query(tables.notes.select());
    const allNotes = Array.isArray(allNotesResult) ? allNotesResult : [];
    const currentNoteIds = new Set(
      allNotes
        .map((n: any) => n.id)
        .filter((id): id is string => typeof id === 'string' && id.length > 0)
    );

    console.log(`SPANN: Found ${currentNoteIds.size} valid notes`);

    // Get all embeddings
    const allEmbeddingRowsResult = this.storeRef.query(tables.embeddings.select());
    const allEmbeddingRows = Array.isArray(allEmbeddingRowsResult) ? allEmbeddingRowsResult : [];
    
    console.log(`SPANN: Found ${allEmbeddingRows.length} total embeddings`);

    // Find stale embeddings
    const staleEmbeddings = allEmbeddingRows.filter((embedding: any) => {
      const noteId = embedding.noteId;
      const isValid = typeof noteId === 'string' && noteId.length > 0;
      const noteExists = isValid && currentNoteIds.has(noteId);
      
      if (!isValid) {
        console.log(`SPANN: Found invalid noteId in embedding:`, noteId);
        return true; // Remove invalid embeddings
      }
      
      if (!noteExists) {
        console.log(`SPANN: Found stale embedding for deleted note:`, noteId);
        return true; // Remove stale embeddings
      }
      
      return false;
    });

    console.log(`SPANN: Found ${staleEmbeddings.length} stale embeddings to remove`);

    // Remove stale embeddings
    const errors: string[] = [];
    let removedCount = 0;

    for (const embedding of staleEmbeddings) {
      try {
        const noteId = embedding.noteId;
        if (typeof noteId === 'string' && noteId.length > 0) {
          console.log(`SPANN: Removing stale embedding for noteId: ${noteId}`);
          this.storeRef.commit(events.embeddingRemoved({ noteId }));
          removedCount++;
        } else {
          console.log(`SPANN: Skipping embedding with invalid noteId:`, noteId);
        }
      } catch (error) {
        const errorMsg = `Failed to remove embedding for noteId ${embedding.noteId}: ${error}`;
        console.error(`SPANN: ${errorMsg}`);
        errors.push(errorMsg);
      }
    }

    // Verify cleanup worked
    const afterCleanupResult = this.storeRef.query(tables.embeddings.select());
    const afterCleanupRows = Array.isArray(afterCleanupResult) ? afterCleanupResult : [];
    console.log(`SPANN: After cleanup: ${afterCleanupRows.length} embeddings remain`);

    // Double-check for any remaining stale embeddings
    const remainingStale = afterCleanupRows.filter((embedding: any) => {
      const noteId = embedding.noteId;
      return !(typeof noteId === 'string' && noteId.length > 0 && currentNoteIds.has(noteId));
    });

    if (remainingStale.length > 0) {
      console.warn(`SPANN: Warning - ${remainingStale.length} stale embeddings still remain after cleanup`);
      remainingStale.forEach(embedding => {
        console.warn(`SPANN: Remaining stale embedding:`, embedding.noteId);
      });
    }

    console.log(`SPANN: Force cleanup complete. Removed ${removedCount} stale embeddings, ${errors.length} errors`);
    
    return {
      removed: removedCount,
      errors,
      totalBefore: allEmbeddingRows.length,
      totalAfter: afterCleanupRows.length,
      remainingStale: remainingStale.length
    };
  }

  /**
   * The new, authoritative "sync and rebuild" process with enhanced cleanup.
   */
  public async buildIndex() {
    if (!this.storeRef || !this.isReady) {
      throw new Error("SpannSearchService is not ready.");
    }
    console.log("SPANN: Starting enhanced sync and rebuild process.");

    // --- Phase 1: Enhanced Stale Embedding Cleanup ---
    console.log("SPANN: Phase 1 - Enhanced stale embedding cleanup");
    
    const cleanupResult = await this.forceCleanupStaleEmbeddings();
    
    if (cleanupResult.errors.length > 0) {
      console.warn(`SPANN: Cleanup completed with ${cleanupResult.errors.length} errors:`, cleanupResult.errors);
    }
    
    console.log(`SPANN: Cleanup summary - Removed: ${cleanupResult.removed}, Before: ${cleanupResult.totalBefore}, After: ${cleanupResult.totalAfter}`);
    
    if (cleanupResult.remainingStale > 0) {
      console.warn(`SPANN: Warning - ${cleanupResult.remainingStale} stale embeddings still remain. This may cause phantom search results.`);
    }

    // --- Phase 2: Sync Current Notes ---
    console.log("SPANN: Phase 2 - Syncing current notes");
    
    const allNotesResult = this.storeRef.query(tables.notes.select());
    const allNotes = Array.isArray(allNotesResult) ? allNotesResult : [];
    
    console.log(`SPANN: Syncing embeddings for ${allNotes.length} current notes.`);
    let syncedCount = 0;
    for (const note of allNotes) {
      try {
        if (typeof note.id === 'string' && note.id.length > 0 && typeof note.title === 'string') {
          await this.addOrUpdateNote(note.id, note.title, note.content);
          syncedCount++;
        } else {
          console.warn(`SPANN: Skipping note with invalid data:`, { id: note.id, title: note.title });
        }
      } catch (error) {
        console.error(`Failed to sync note ${note.id}:`, error);
      }
    }

    // --- Phase 3: Rebuild the SPANN Index ---
    console.log("SPANN: Phase 3 - Rebuilding SPANN index");
    
    const allEmbeddingsResult = this.storeRef.query(tables.embeddings.select());
    const allEmbeddings = Array.isArray(allEmbeddingsResult) ? allEmbeddingsResult : [];
    const minEmbeddings = 3;
    
    console.log(`SPANN: Found ${allEmbeddings.length} embeddings for index building`);
    
    if (!allEmbeddings || allEmbeddings.length < minEmbeddings) {
      const message = `Not enough embeddings (${allEmbeddings?.length || 0}) to build index. Need at least ${minEmbeddings}.`;
      console.error(message);
      throw new Error(message);
    }

    // Clear old cluster data
    this.storeRef.commit(events.embeddingIndexCleared({}));
    
    // Select centroids (random sampling from the clean data)
    const sampledEmbeddings: any[] = [];
    const usedIndices = new Set<number>();
    const targetCount = Math.min(this.config.numClusters, allEmbeddings.length);
    while (sampledEmbeddings.length < targetCount) {
      const randomIndex = Math.floor(Math.random() * allEmbeddings.length);
      if (!usedIndices.has(randomIndex)) {
        sampledEmbeddings.push(allEmbeddings[randomIndex]);
        usedIndices.add(randomIndex);
      }
    }

    // Commit new centroids
    sampledEmbeddings.forEach((embedding, i) => {
      this.storeRef.commit(events.embeddingClusterCreated({
        id: i,
        vecData: embedding.vecData,
        vecDim: embedding.vecDim,
        createdAt: new Date().toISOString(),
      }));
    });

    // Reload centroids and build in-memory index
    await this.loadCentroidsFromDB();
    if (!this.centroidIndex) throw new Error("Failed to build centroid index after sync.");

    // Assign each embedding to its nearest cluster
    console.log(`SPANN: Assigning ${allEmbeddings.length} embeddings to ${this.centroids.length} clusters.`);
    const assignments = new Map<number, string[]>();
    for (const embedding of allEmbeddings) {
      const vector = blobToVec(embedding.vecData, embedding.vecDim);
      const [nearestCentroid] = this.centroidIndex.searchKNN(vector, 1);
      if (nearestCentroid) {
        if (!assignments.has(nearestCentroid.id)) {
          assignments.set(nearestCentroid.id, []);
        }
        assignments.get(nearestCentroid.id)!.push(embedding.noteId);
      }
    }
    
    // Commit cluster assignments
    for (const [clusterId, noteIds] of assignments.entries()) {
      this.storeRef.commit(events.embeddingsAssignedToCluster({ clusterId, noteIds }));
    }

    // Final verification
    const finalEmbeddingCount = this.storeRef.query(tables.embeddings.count()) || 0;
    const finalNoteCount = this.storeRef.query(tables.notes.count()) || 0;
    
    console.log(`SPANN: Final verification - Notes: ${finalNoteCount}, Embeddings: ${finalEmbeddingCount}`);
    
    if (finalEmbeddingCount > finalNoteCount) {
      console.warn(`SPANN: Warning - More embeddings (${finalEmbeddingCount}) than notes (${finalNoteCount}). Some stale data may remain.`);
    }

    console.log(`SPANN: Enhanced sync and rebuild complete. Synced ${syncedCount} notes, cleaned ${cleanupResult.removed} stale embeddings.`);
    return this.centroids.length;
  }

  /**
   * Performs the two-phase hybrid search.
   */
  public async search(query: string, k = 10): Promise<SearchResult[]> {
    try {
      await this.initialize();
      
      if (!query.trim() || !this.centroidIndex) {
        return [];
      }

      // Phase 1: In-Memory Search
      // Find the most relevant clusters by searching the small centroid index
      const { vector: queryVector } = await embeddingService.embed([`search_query: ${query}`]);
      const normalizedQueryVector = l2Normalize(queryVector);
      
      const candidateClusters = this.centroidIndex.searchKNN(normalizedQueryVector, this.config.searchProbeCount);
      
      if (candidateClusters.length === 0) return [];

      // Phase 2: Disk Search
      // For each candidate cluster, fetch all embeddings with that clusterId
      const allCandidateEmbeddings: any[] = [];
      
      for (const cluster of candidateClusters) {
        const embeddingsForCluster = this.storeRef.query(
          tables.embeddings.select().where({ clusterId: cluster.id })
        );
        if (Array.isArray(embeddingsForCluster)) {
          allCandidateEmbeddings.push(...embeddingsForCluster);
        }
      }

      if (allCandidateEmbeddings.length === 0) {
        return [];
      }
      
      // Perform a final, exact search on the small subset of retrieved vectors
      const finalResults: SearchResult[] = [];
      for (const embedding of allCandidateEmbeddings) {
        const vector = blobToVec(embedding.vecData, embedding.vecDim);
        const normalizedVector = l2Normalize(vector);
        const score = this.centroidIndex.similarityFunction(normalizedQueryVector, normalizedVector);
        finalResults.push({
          noteId: embedding.noteId,
          title: embedding.title,
          content: embedding.content,
          score
        });
      }

      return finalResults
        .sort((a, b) => b.score - a.score)
        .slice(0, k);
    } catch (error) {
      console.error('Search failed:', error);
      toast.error('Search failed');
      return [];
    }
  }

  /**
   * Adds or updates a single note's embedding. For this simplified model,
   * we add it to the database without a cluster assignment. It will be
   * properly clustered during the next full buildIndex run.
   */
  public async addOrUpdateNote(noteId: string, title: string, content: Block[]) {
    try {
      await this.initialize();
      
      const textContent = blocksToText(content);
      if (!textContent.trim()) {
        this.removeNote(noteId);
        return;
      }

      const { vector } = await embeddingService.embed([textContent]);
      const normalizedVector = l2Normalize(vector);
      
      if (this.storeRef) {
        this.storeRef.commit(events.noteEmbedded({
          noteId,
          title,
          content: textContent,
          vecData: vecToBlob(normalizedVector),
          vecDim: normalizedVector.length,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          clusterId: null // This embedding is now "unclustered"
        }));
      }
    } catch (error) {
      console.error('Failed to embed note:', error);
      toast.error('Failed to generate embedding for note');
    }
  }

  public removeNote(noteId: string) {
    try {
      if (this.storeRef && typeof noteId === 'string' && noteId.length > 0) {
        console.log(`SPANN: Removing embedding for noteId: ${noteId}`);
        this.storeRef.commit(events.embeddingRemoved({ noteId }));
      } else {
        console.warn('SpannSearchService: Cannot remove embedding - invalid noteId or no store reference');
      }
    } catch (error) {
      console.error('Failed to remove embedding:', error);
    }
  }

  /**
   * @deprecated Use buildIndex() instead for authoritative sync and rebuild
   */
  public async syncAllNotes(notes: Array<{ id: string; title: string; content: Block[] }>) {
    console.warn('syncAllNotes is deprecated. Use buildIndex() for authoritative sync and rebuild.');
    
    try {
      await this.initialize();
      
      console.log(`SpannSearchService: Syncing ${notes.length} notes`);
      
      let successCount = 0;
      let errorCount = 0;
      
      for (const note of notes) {
        try {
          await this.addOrUpdateNote(note.id, note.title, note.content);
          successCount++;
        } catch (error) {
          errorCount++;
          console.error(`Failed to sync note ${note.id}:`, error);
        }
      }
      
      console.log(`SpannSearchService: Sync complete, ${successCount} embeddings created, ${errorCount} errors`);
      return successCount;
    } catch (error) {
      console.error('Failed to sync notes:', error);
      toast.error('Failed to synchronize embeddings');
      return 0;
    }
  }

  public getEmbeddingCount(): number {
    if (!this.storeRef) return 0;
    try {
      return this.storeRef.query(tables.embeddings.count()) || 0;
    } catch (error) {
      console.error('Failed to get embedding count:', error);
      return 0;
    }
  }

  public getStoredEmbeddingCount(): number {
    return this.getEmbeddingCount();
  }

  public getCentroidCount(): number {
    if (!this.storeRef) return 0;
    try {
      return this.storeRef.query(tables.embeddingClusters.count()) || 0;
    } catch (error) {
      console.error('Failed to get centroid count:', error);
      return 0;
    }
  }

  public isIndexBuilt(): boolean {
    return this.centroidIndex !== null && this.centroids.length > 0;
  }
}

// Export a singleton instance to be used throughout the app
export const spannSearchService = new SpannSearchService();
