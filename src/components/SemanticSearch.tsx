
import React, { useState, useCallback, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Search, Loader2, Database, Building, Trash2, HardDrive } from 'lucide-react';
import { useDebouncedCallback } from '@/hooks/useDebouncedCallback';
import { spannSearchService } from '@/lib/embedding/SpannSearchService';
import { useActiveNoteId, useNotes } from '@/hooks/useLiveStore';
import { useEmbeddings, useEmbeddingCount, useEmbeddingClustersCount } from '@/hooks/useEmbeddings';
import { toast } from 'sonner';

interface SearchResult {
  noteId: string;
  title: string;
  content: string;
  score: number;
}

export function SemanticSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [isSyncingAndRebuilding, setIsSyncingAndRebuilding] = useState(false);
  const [isCleaningUp, setIsCleaningUp] = useState(false);
  const [snapshotInfo, setSnapshotInfo] = useState<{ count: number; latestDate?: Date; totalSize?: number }>({ count: 0 });
  const [, setActiveNoteId] = useActiveNoteId();
  const notes = useNotes();
  
  // Use LiveStore reactive queries
  const embeddings = useEmbeddings();
  const embeddingCount = useEmbeddingCount();
  const clustersCount = useEmbeddingClustersCount();

  // Load snapshot info on mount and after operations
  const loadSnapshotInfo = useCallback(async () => {
    try {
      const info = await spannSearchService.getSnapshotInfo();
      setSnapshotInfo(info);
    } catch (error) {
      console.error('Failed to load snapshot info:', error);
    }
  }, []);

  useEffect(() => {
    loadSnapshotInfo();
  }, [loadSnapshotInfo]);

  const handleSearch = useDebouncedCallback(async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const searchResults = await spannSearchService.search(searchQuery, 10);
      setResults(searchResults);
    } catch (error) {
      console.error('Search failed:', error);
      toast.error('Search failed');
    } finally {
      setIsSearching(false);
    }
  }, 500);

  const handleForceCleanup = useCallback(async () => {
    setIsCleaningUp(true);
    try {
      const result = await spannSearchService.forceCleanupStaleEmbeddings();
      
      if (result.removed > 0) {
        toast.success(`Cleanup complete! Removed ${result.removed} stale embeddings`);
      } else {
        toast.success('No stale embeddings found - database is clean');
      }
      
      if (result.errors.length > 0) {
        toast.error(`Cleanup had ${result.errors.length} errors. Check console for details.`);
      }
      
      console.log('Cleanup result:', result);
      await loadSnapshotInfo(); // Refresh snapshot info
    } catch (error) {
      console.error('Force cleanup failed:', error);
      toast.error('Failed to cleanup stale embeddings');
    } finally {
      setIsCleaningUp(false);
    }
  }, [loadSnapshotInfo]);

  const handleSyncAndRebuildIndex = useCallback(async () => {
    setIsSyncingAndRebuilding(true);
    try {
      const centroidCount = await spannSearchService.buildIndex();
      toast.success(`Sync & rebuild complete! Built index with ${centroidCount} clusters`);
      await loadSnapshotInfo(); // Refresh snapshot info after rebuild
    } catch (error) {
      console.error('Sync and rebuild failed:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to sync and rebuild index');
    } finally {
      setIsSyncingAndRebuilding(false);
    }
  }, [loadSnapshotInfo]);

  const handleSelectNote = (noteId: string) => {
    setActiveNoteId(noteId);
  };

  const isIndexBuilt = spannSearchService.isIndexBuilt();
  const minEmbeddingsRequired = 3;
  const hasStaleData = embeddingCount > notes.length;

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="p-4 flex flex-col h-full">
      <div className="space-y-3 mb-4">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Semantic search..."
            className="pl-8"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              handleSearch(e.target.value);
            }}
          />
        </div>
        
        <div className="flex flex-col space-y-2">
          <Button
            variant={isIndexBuilt ? "outline" : "default"}
            size="sm"
            className="w-full"
            onClick={handleSyncAndRebuildIndex}
            disabled={isSyncingAndRebuilding || embeddingCount < minEmbeddingsRequired}
          >
            {isSyncingAndRebuilding ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Building className="h-4 w-4 mr-2" />
            )}
            {isIndexBuilt ? 'Sync & Rebuild Index' : 'Sync & Build Index'}
          </Button>
          
          {hasStaleData && (
            <Button
              variant="outline"
              size="sm"
              className="w-full text-destructive hover:bg-destructive/10"
              onClick={handleForceCleanup}
              disabled={isCleaningUp}
            >
              {isCleaningUp ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4 mr-2" />
              )}
              Force Cleanup Stale Data
            </Button>
          )}
          
          {embeddingCount < minEmbeddingsRequired && (
            <p className="text-xs text-muted-foreground text-center">
              Need at least {minEmbeddingsRequired} notes to build the search index
            </p>
          )}
          
          {/* Display embedding and index status */}
          <div className="flex flex-col space-y-1 text-xs text-muted-foreground">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Database className="h-3 w-3 mr-1" />
                <span>{embeddingCount} embeddings</span>
              </div>
              <div className="flex items-center">
                <Building className="h-3 w-3 mr-1" />
                <span>{clustersCount} clusters</span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span>{notes.length} notes</span>
              {hasStaleData && (
                <Badge variant="destructive" className="text-xs">
                  Stale Data Detected
                </Badge>
              )}
            </div>
            
            {/* Graph persistence status */}
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <HardDrive className="h-3 w-3 mr-1" />
                <span>{snapshotInfo.count} snapshots</span>
              </div>
              {snapshotInfo.totalSize && (
                <span>{formatBytes(snapshotInfo.totalSize)}</span>
              )}
            </div>
            
            {snapshotInfo.latestDate && (
              <div className="text-center text-xs">
                Last: {snapshotInfo.latestDate.toLocaleString()}
              </div>
            )}
            
            <div className="text-center">
              {isIndexBuilt ? (
                <Badge variant="secondary" className="text-xs">Index Ready</Badge>
              ) : (
                <Badge variant="outline" className="text-xs">Index Not Built</Badge>
              )}
            </div>
          </div>
        </div>
      </div>

      {isSearching && (
        <div className="flex justify-center py-4">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      )}

      <ScrollArea className="flex-1">
        <div className="space-y-2">
          {results.map((result) => (
            <div
              key={result.noteId}
              className="p-3 rounded-md border border-transparent hover:border-primary/20 hover:bg-primary/10 cursor-pointer transition-colors"
              onClick={() => handleSelectNote(result.noteId)}
            >
              <div className="flex justify-between items-center mb-1">
                <div className="font-medium text-sm truncate">{result.title}</div>
                <Badge variant="secondary" className="text-xs">
                  {Math.round(result.score * 100)}%
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground line-clamp-2">
                {result.content.slice(0, 100)}...
              </p>
            </div>
          ))}
          {query && !isSearching && results.length === 0 && (
            <div className="text-center text-muted-foreground py-4">
              <p className="text-sm">No results found</p>
              {!isIndexBuilt && (
                <p className="text-xs mt-1">Build the search index first for better results</p>
              )}
              {hasStaleData && (
                <p className="text-xs mt-1 text-destructive">Try cleaning up stale data first</p>
              )}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
