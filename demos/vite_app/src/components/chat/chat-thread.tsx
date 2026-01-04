import { useRef, useEffect, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { UserMessage } from './user-message';
import { AssistantMessage } from './assistant-message';
import { FrontendToolPrompt } from '@/components/agent/frontend-tool-prompt';
import { Skeleton } from '@/components/ui/skeleton';
import type { ConversationTurn } from '@/hooks/use-conversation';
import type { PendingFrontendTool } from '@/lib/parsers/types';

interface ChatThreadProps {
  turns: ConversationTurn[];
  isLoadingHistory: boolean;
  hasMore: boolean;
  isAwaitingTools: boolean;
  pendingFrontendTools?: PendingFrontendTool[];
  onLoadMore: () => void;
  onToolResponse: (tool: PendingFrontendTool, response: string) => void;
  className?: string;
}

/**
 * Scrollable chat thread with infinite scroll for loading older messages.
 */
export function ChatThread({
  turns,
  isLoadingHistory,
  hasMore,
  isAwaitingTools,
  pendingFrontendTools,
  onLoadMore,
  onToolResponse,
  className,
}: ChatThreadProps) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const topSentinelRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const previousTurnsLengthRef = useRef(turns.length);
  const isScrollingToBottomRef = useRef(false);

  const scrollToBottom = useCallback(() => {
    if (bottomRef.current && !isScrollingToBottomRef.current) {
      isScrollingToBottomRef.current = true;
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
      setTimeout(() => {
        isScrollingToBottomRef.current = false;
      }, 100);
    }
  }, []);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    if (turns.length > previousTurnsLengthRef.current) {
      // New message added - scroll to bottom
      scrollToBottom();
    }
    previousTurnsLengthRef.current = turns.length;
  }, [turns.length, scrollToBottom]);

  // Also scroll when streaming content updates
  useEffect(() => {
    const streamingTurn = turns.find(t => t.isStreaming);
    if (streamingTurn) {
      scrollToBottom();
    }
  }, [turns, scrollToBottom]);

  // Intersection observer for infinite scroll (load more on scroll to top)
  useEffect(() => {
    if (!topSentinelRef.current || !hasMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting && !isLoadingHistory && hasMore) {
          onLoadMore();
        }
      },
      {
        root: scrollContainerRef.current,
        rootMargin: '100px 0px 0px 0px', // Trigger 100px before reaching top
        threshold: 0,
      }
    );

    observer.observe(topSentinelRef.current);
    return () => observer.disconnect();
  }, [hasMore, isLoadingHistory, onLoadMore]);

  return (
    <div
      ref={scrollContainerRef}
      className={cn(
        'flex-1 overflow-y-auto px-4 py-4',
        className
      )}
    >
      {/* Top sentinel for infinite scroll */}
      <div ref={topSentinelRef} className="h-1" />
      
      {/* Loading indicator for history */}
      {isLoadingHistory && (
        <div className="mb-4 space-y-3">
          <Skeleton className="h-12 w-3/4" />
          <Skeleton className="ml-auto h-8 w-1/2" />
          <Skeleton className="h-16 w-2/3" />
        </div>
      )}
      
      {/* Load more indicator */}
      {hasMore && !isLoadingHistory && (
        <div className="mb-4 text-center">
          <button
            onClick={onLoadMore}
            className="text-xs text-zinc-400 hover:text-zinc-600 dark:text-zinc-500 dark:hover:text-zinc-300"
          >
            Load older messages
          </button>
        </div>
      )}

      {/* Empty state */}
      {turns.length === 0 && !isLoadingHistory && (
        <div className="flex h-full items-center justify-center">
          <p className="text-zinc-400 dark:text-zinc-500">
            Start a conversation by typing a message below.
          </p>
        </div>
      )}

      {/* Conversation turns */}
      <div className="space-y-4">
        {turns.map((turn) => (
          <div key={turn.id} className="space-y-3">
            {/* User message */}
            <UserMessage message={turn.userMessage} />
            
            {/* Assistant response */}
            <AssistantMessage
              response={turn.assistantResponse}
              streamingState={turn.streamingState}
              nodes={turn.nodes}
              isStreaming={turn.isStreaming}
              usage={turn.usage}
              totalSteps={turn.totalSteps}
              timestamp={turn.completedAt}
            />
          </div>
        ))}
      </div>

      {/* Frontend tool prompts */}
      {isAwaitingTools && pendingFrontendTools && pendingFrontendTools.length > 0 && (
        <div className="mt-4 space-y-2">
          {pendingFrontendTools.map((tool) => (
            <FrontendToolPrompt
              key={tool.tool_use_id}
              tool={tool}
              onSubmit={(response) => onToolResponse(tool, response)}
            />
          ))}
        </div>
      )}

      {/* Bottom anchor for auto-scroll */}
      <div ref={bottomRef} className="h-1" />
    </div>
  );
}

