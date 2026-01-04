import { useEffect, useState, useCallback } from 'react';
import { fetchAgentSessions, type AgentSession } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { MessageSquare, PanelLeftClose, PanelLeft, Plus, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ConversationSidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  onSelectConversation: (agentUuid: string) => void;
  onNewConversation: () => void;
  currentAgentUuid?: string;
  /** Increment to trigger a refresh of the sessions list */
  refreshTrigger?: number;
}

/**
 * Format a date string as a relative time (e.g., "2 hours ago", "Yesterday")
 */
function formatRelativeTime(dateString: string | null): string {
  if (!dateString) return '';
  
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  
  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays}d ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)}w ago`;
  
  return date.toLocaleDateString();
}

/**
 * Sidebar showing list of past agent conversations.
 * Collapsible with toggle button.
 */
export function ConversationSidebar({
  isOpen,
  onToggle,
  onSelectConversation,
  onNewConversation,
  currentAgentUuid,
  refreshTrigger,
}: ConversationSidebarProps) {
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [total, setTotal] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const LIMIT = 50;

  const loadSessions = useCallback(async (loadOffset: number = 0, append: boolean = false) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetchAgentSessions(LIMIT, loadOffset);
      if (append) {
        setSessions(prev => [...prev, ...response.sessions]);
      } else {
        setSessions(response.sessions);
      }
      setTotal(response.total);
      setOffset(loadOffset + response.sessions.length);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load sessions');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initial load and refresh when trigger changes
  useEffect(() => {
    loadSessions(0, false);
  }, [loadSessions, refreshTrigger]);

  const handleLoadMore = () => {
    if (!isLoading && sessions.length < total) {
      loadSessions(offset, true);
    }
  };

  const hasMore = sessions.length < total;

  // Collapsed state - just show toggle button
  if (!isOpen) {
    return (
      <div className="flex h-full w-12 flex-col border-r border-sidebar-border bg-sidebar">
        <div className="flex h-14 items-center justify-center border-b border-sidebar-border">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggle}
            className="text-sidebar-foreground hover:bg-sidebar-accent"
            title="Open sidebar"
          >
            <PanelLeft className="h-5 w-5" />
          </Button>
        </div>
        <div className="flex flex-1 flex-col items-center gap-2 pt-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={onNewConversation}
            className="text-sidebar-foreground hover:bg-sidebar-accent"
            title="New conversation"
          >
            <Plus className="h-5 w-5" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-64 min-h-0 flex-col overflow-hidden border-r border-sidebar-border bg-sidebar">
      {/* Header */}
      <div className="flex h-14 items-center justify-between border-b border-sidebar-border px-3">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-sidebar-foreground" />
          <span className="font-medium text-sidebar-foreground">Conversations</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className="h-8 w-8 text-sidebar-foreground hover:bg-sidebar-accent"
          title="Close sidebar"
        >
          <PanelLeftClose className="h-4 w-4" />
        </Button>
      </div>

      {/* New Chat Button */}
      <div className="border-b border-sidebar-border p-2">
        <Button
          variant="outline"
          onClick={onNewConversation}
          className="w-full justify-start gap-2 border-sidebar-border bg-transparent text-sidebar-foreground hover:bg-sidebar-accent"
        >
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </div>

      {/* Sessions List */}
      <ScrollArea className="flex-1 min-h-0">
        {error && (
          <div className="p-3 text-sm text-red-500">{error}</div>
        )}
        
        <div className="flex flex-col gap-0.5 p-2">
          {sessions.map((session) => (
            <button
              key={session.agent_uuid}
              onClick={() => onSelectConversation(session.agent_uuid)}
              className={cn(
                'flex w-full flex-col items-start gap-0.5 rounded-md px-3 py-2 text-left text-sm transition-colors',
                'hover:bg-sidebar-accent',
                currentAgentUuid === session.agent_uuid
                  ? 'bg-sidebar-accent text-sidebar-accent-foreground'
                  : 'text-sidebar-foreground'
              )}
            >
              <span className="line-clamp-1 font-medium">
                {session.title || 'Untitled'}
              </span>
              <span className="text-xs text-sidebar-foreground/60">
                {formatRelativeTime(session.updated_at)}
              </span>
            </button>
          ))}
        </div>

        {/* Load More */}
        {hasMore && (
          <div className="p-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleLoadMore}
              disabled={isLoading}
              className="w-full text-sidebar-foreground hover:bg-sidebar-accent"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Loading...
                </>
              ) : (
                'Load more'
              )}
            </Button>
          </div>
        )}

        {/* Initial loading state */}
        {isLoading && sessions.length === 0 && (
          <div className="flex items-center justify-center p-4">
            <Loader2 className="h-6 w-6 animate-spin text-sidebar-foreground/60" />
          </div>
        )}

        {/* Empty state */}
        {!isLoading && sessions.length === 0 && !error && (
          <div className="p-4 text-center text-sm text-sidebar-foreground/60">
            No conversations yet
          </div>
        )}
      </ScrollArea>
    </div>
  );
}

