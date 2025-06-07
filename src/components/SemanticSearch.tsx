
import React, { useState, useCallback, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Search, Loader2, Zap, Database, Building } from 'lucide-react';
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
  const [isLoading, setIsLoading] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const [isBuildingIndex, setIsBuildingIndex] = useState(false);
  const [syncCount, setSyncCount] = useState(0);
  const [, setActiveNoteId] = useActiveNoteId();
  const notes = useNotes();
  
  // Use LiveStore reactive queries
  const embeddings = useEmbeddings();
  const embeddingCount = useEmbeddingCount();
  const clustersCount = useEmbeddingClustersCount();
  
  // Update sync count on mount and when embeddings change
  useEffect(() => {
    setSyncCount(spannSearchService.getEmbeddingCount());
  }, [embeddings]);

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

  const handleSyncEmbeddings = useCallback(async () => {
    setIsLoading(true);
    try {
      const count = await spannSearchService.syncAllNotes(notes);
      setSyncCount(count);
      toast.success(`Synchronized ${count} note embeddings`);
    } catch (error) {
      console.error('Sync failed:', error);
      toast.error('Failed to sync embeddings');
    } finally {
      setIsLoading(false);
    }
  }, [notes]);

  const handleBuildIndex = useCallback(async () => {
    setIsBuildingIndex(true);
    try {
      const centroidCount = await spannSearchService.buildIndex();
      toast.success(`Built SPANN index with ${centroidCount} clusters`);
    } catch (error) {
      console.error('Index build failed:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to build index');
    } finally {
      setIsBuildingIndex(false);
    }
  }, []);

  const handleSelectNote = (noteId: string) => {
    setActiveNoteId(noteId);
  };

  const isIndexBuilt = spannSearchService.isIndexBuilt();

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
            variant="outline"
            size="sm"
            className="w-full"
            onClick={handleSyncEmbeddings}
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Zap className="h-4 w-4 mr-2" />
            )}
            Sync Embeddings
          </Button>
          
          <Button
            variant={isIndexBuilt ? "outline" : "default"}
            size="sm"
            className="w-full"
            onClick={handleBuildIndex}
            disabled={isBuildingIndex || embeddingCount < 10}
          >
            {isBuildingIndex ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Building className="h-4 w-4 mr-2" />
            )}
            {isIndexBuilt ? 'Rebuild Index' : 'Build Index'}
          </Button>
          
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
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
