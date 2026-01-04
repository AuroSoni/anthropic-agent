import { cn } from '@/lib/utils';
import { NodeListRenderer } from '@/components/agent/node-renderer';
import { StreamingIndicator } from '@/components/agent/streaming-indicator';
import type { AgentNode } from '@/lib/parsers/types';
import type { AgentState } from '@/lib/agent-stream';

interface AssistantMessageProps {
  /** Plain text response for completed messages */
  response?: string | null;
  /** Streaming state for live updates */
  streamingState?: AgentState;
  /** Pre-parsed nodes from history (converted from messages array) */
  nodes?: AgentNode[];
  /** Whether this message is currently streaming */
  isStreaming?: boolean;
  /** Completion stats */
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
  totalSteps?: number;
  timestamp?: string;
  className?: string;
}

/**
 * Assistant message bubble - left-aligned with rich content rendering.
 */
export function AssistantMessage({
  response,
  streamingState,
  nodes,
  isStreaming,
  usage,
  totalSteps,
  timestamp,
  className,
}: AssistantMessageProps) {
  // Determine which nodes to render
  const displayNodes = streamingState?.nodes ?? nodes ?? [];
  const showStreamingIndicator = isStreaming && (!displayNodes || displayNodes.length === 0);
  
  return (
    <div className={cn('flex justify-start', className)}>
      <div className="max-w-[85%] space-y-1">
        <div className="rounded-2xl rounded-bl-md bg-zinc-100 px-4 py-2.5 shadow-sm dark:bg-zinc-800">
          {displayNodes.length > 0 ? (
            <div className="text-sm text-zinc-900 dark:text-zinc-100">
              <NodeListRenderer nodes={displayNodes} />
            </div>
          ) : response ? (
            <p className="whitespace-pre-wrap text-sm text-zinc-900 dark:text-zinc-100">
              {response}
            </p>
          ) : showStreamingIndicator ? (
            <StreamingIndicator />
          ) : (
            <p className="text-sm italic text-zinc-400 dark:text-zinc-500">
              No response
            </p>
          )}
          
          {isStreaming && displayNodes.length > 0 && (
            <div className="mt-2">
              <StreamingIndicator />
            </div>
          )}
        </div>
        
        {/* Metadata footer */}
        <div className="flex items-center gap-2 text-xs text-zinc-400 dark:text-zinc-500">
          {timestamp && <span>{formatTimestamp(timestamp)}</span>}
          {usage && (
            <span>
              {usage.input_tokens} in / {usage.output_tokens} out
            </span>
          )}
          {totalSteps !== undefined && totalSteps > 0 && (
            <span>{totalSteps} step{totalSteps !== 1 ? 's' : ''}</span>
          )}
        </div>
      </div>
    </div>
  );
}

function formatTimestamp(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

