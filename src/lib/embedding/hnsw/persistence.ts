
import { HNSW } from './main';
import { encodeGraph, decodeGraph } from './serialize';
import { createChecksum } from '../../utils/checksum';
import { events, tables } from '../../../livestore/schema';

export class HnswPersistence {
  private static instance: HnswPersistence | null = null;
  private storeRef: any = null;

  static getInstance(): HnswPersistence {
    if (!HnswPersistence.instance) {
      HnswPersistence.instance = new HnswPersistence();
    }
    return HnswPersistence.instance;
  }

  setStore(store: any) {
    this.storeRef = store;
  }

  async persistGraph(index: HNSW): Promise<void> {
    if (!this.storeRef) {
      console.warn('HnswPersistence: No store reference available');
      return;
    }

    try {
      console.log('HnswPersistence: Starting graph persistence');
      
      // 1. Serialize the graph
      const data = encodeGraph(index);
      console.log(`HnswPersistence: Serialized graph to ${data.length} bytes`);

      // 2. Create filename with timestamp
      const fileName = `graphs/snap-${Date.now()}.bin`;
      
      // 3. Write to OPFS using async API
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle('graphs', { create: true });
      const file = await dir.getFileHandle(fileName.split('/').at(-1)!, { create: true });
      
      // Use writable stream instead of sync access handle
      const writable = await file.createWritable();
      await writable.write(data);
      await writable.close();
      
      console.log(`HnswPersistence: Wrote graph to OPFS: ${fileName}`);

      // 4. Generate checksum and emit event
      const checksum = await createChecksum(data);
      const createdAt = new Date(); // Use Date object to match event schema
      
      this.storeRef.commit(events.hnswGraphSnapshotCreated({ 
        fileName, 
        checksum, 
        createdAt 
      }));
      
      console.log(`HnswPersistence: Graph snapshot saved with checksum ${checksum}`);
    } catch (error) {
      console.error('HnswPersistence: Failed to persist graph:', error);
      throw error;
    }
  }

  async loadLatestGraph(): Promise<HNSW | null> {
    if (!this.storeRef) {
      console.warn('HnswPersistence: No store reference available for loading');
      return null;
    }

    try {
      console.log('HnswPersistence: Loading latest graph snapshot');
      
      // Get the most recent snapshot
      const snapshots = this.storeRef.query(
        tables.hnswSnapshots.orderBy('createdAt', 'desc').limit(1)
      );
      
      if (!Array.isArray(snapshots) || snapshots.length === 0) {
        console.log('HnswPersistence: No graph snapshots found');
        return null;
      }

      const meta = snapshots[0];
      console.log(`HnswPersistence: Found snapshot: ${meta.fileName}`);

      // Read file from OPFS
      const root = await navigator.storage.getDirectory();
      const fileName = meta.fileName.split('/').at(-1)!;
      const dir = await root.getDirectoryHandle('graphs', { create: false });
      const file = await dir.getFileHandle(fileName, { create: false });
      const fileHandle = await file.getFile();
      const fileData = new Uint8Array(await fileHandle.arrayBuffer());
      
      console.log(`HnswPersistence: Read ${fileData.length} bytes from OPFS`);

      // Verify checksum
      const actualChecksum = await createChecksum(fileData);
      if (actualChecksum !== meta.checksum) {
        console.error('HnswPersistence: Checksum mismatch - data corruption detected');
        throw new Error('Graph checksum mismatch – data corruption');
      }

      // Decode and return graph
      const graph = decodeGraph(fileData);
      console.log(`HnswPersistence: Successfully loaded graph with ${graph.nodes.size} nodes`);
      
      return graph;
    } catch (error) {
      console.error('HnswPersistence: Failed to load graph:', error);
      return null;
    }
  }

  async gcOldSnapshots(keepLast = 2): Promise<number> {
    if (!this.storeRef) {
      console.warn('HnswPersistence: No store reference available for GC');
      return 0;
    }

    try {
      console.log(`HnswPersistence: Starting GC, keeping last ${keepLast} snapshots`);
      
      const snapshots = this.storeRef.query(
        tables.hnswSnapshots.orderBy('createdAt', 'desc')
      );

      if (!Array.isArray(snapshots) || snapshots.length <= keepLast) {
        console.log('HnswPersistence: No snapshots to clean up');
        return 0;
      }

      const toDelete = snapshots.slice(keepLast);
      let deletedCount = 0;

      for (const snapshot of toDelete) {
        try {
          // Remove from LiveStore
          this.storeRef.commit(events.hnswSnapshotDeleted({ fileName: snapshot.fileName }));
          
          // Remove file from OPFS
          const root = await navigator.storage.getDirectory();
          const fileName = snapshot.fileName.split('/').at(-1)!;
          const dir = await root.getDirectoryHandle('graphs', { create: false });
          await dir.removeEntry(fileName);
          
          deletedCount++;
          console.log(`HnswPersistence: Deleted snapshot ${snapshot.fileName}`);
        } catch (error) {
          console.error(`HnswPersistence: Failed to delete snapshot ${snapshot.fileName}:`, error);
        }
      }

      console.log(`HnswPersistence: GC complete, deleted ${deletedCount} snapshots`);
      return deletedCount;
    } catch (error) {
      console.error('HnswPersistence: GC failed:', error);
      return 0;
    }
  }

  async getSnapshotInfo(): Promise<{ count: number; latestDate?: Date; totalSize?: number }> {
    if (!this.storeRef) {
      return { count: 0 };
    }

    try {
      const snapshots = this.storeRef.query(tables.hnswSnapshots.orderBy('createdAt', 'desc'));
      
      if (!Array.isArray(snapshots) || snapshots.length === 0) {
        return { count: 0 };
      }

      const latestDate = new Date(snapshots[0].createdAt);
      
      // Try to get total size from OPFS
      let totalSize = 0;
      try {
        const root = await navigator.storage.getDirectory();
        const dir = await root.getDirectoryHandle('graphs', { create: false });
        
        for (const snapshot of snapshots) {
          try {
            const fileName = snapshot.fileName.split('/').at(-1)!;
            const file = await dir.getFileHandle(fileName, { create: false });
            const fileHandle = await file.getFile();
            totalSize += fileHandle.size;
          } catch {
            // File might not exist, skip
          }
        }
      } catch {
        // Directory might not exist, skip size calculation
      }

      return {
        count: snapshots.length,
        latestDate,
        totalSize: totalSize > 0 ? totalSize : undefined
      };
    } catch (error) {
      console.error('HnswPersistence: Failed to get snapshot info:', error);
      return { count: 0 };
    }
  }
}

export const hnswPersistence = HnswPersistence.getInstance();
