import { useState, useEffect, useRef, type FormEvent, type KeyboardEvent } from 'react';
import { useConversation } from '@/hooks/use-conversation';
import { ChatThread } from '@/components/chat';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Send, Plus } from 'lucide-react';
import { AGENT_TYPES, type AgentType, type UserPrompt } from '@/lib/agent-stream';
import { createToolResult } from '@/lib/frontend-tools';
import type { PendingFrontendTool } from '@/lib/parsers/types';

type PromptFormat = 'text' | 'json';

interface AgentViewerProps {
  /** Callback fired when streaming completes (for sidebar refresh) */
  onStreamComplete?: () => void;
}

/**
 * AgentViewer is the main chat interface container.
 * Displays a conversation thread with user/assistant messages and handles input.
 */
export function AgentViewer({ onStreamComplete }: AgentViewerProps) {
  const {
    turns,
    hasMore,
    isLoadingHistory,
    isStreaming,
    isAwaitingTools,
    pendingFrontendTools,
    error,
    agentUuid,
    title,
    sendMessage,
    submitToolResults,
    loadMore,
    startNewConversation,
  } = useConversation();

  const [prompt, setPrompt] = useState('');
  const [agentType, setAgentType] = useState<AgentType>('agent_frontend_tools');
  const [promptFormat, setPromptFormat] = useState<PromptFormat>('text');
  const [promptError, setPromptError] = useState<string | null>(null);

  // Track previous streaming state to detect completion
  const wasStreamingRef = useRef(false);

  // Notify parent when streaming completes (for sidebar refresh)
  useEffect(() => {
    if (wasStreamingRef.current && !isStreaming) {
      onStreamComplete?.();
    }
    wasStreamingRef.current = isStreaming;
  }, [isStreaming, onStreamComplete]);

  /**
   * Handle frontend tool response from user interaction.
   */
  const handleToolResponse = (respondedTool: PendingFrontendTool, response: string) => {
    const results = pendingFrontendTools?.map(tool => 
      tool.tool_use_id === respondedTool.tool_use_id 
        ? createToolResult(tool, response)
        : createToolResult(tool, 'skipped')
    ) ?? [];
    
    if (pendingFrontendTools?.length === 1) {
      submitToolResults([createToolResult(respondedTool, response)]);
    } else {
      submitToolResults(results);
    }
  };

  const parsePrompt = (): UserPrompt | null => {
    const trimmed = prompt.trim();
    if (!trimmed) return null;

    if (promptFormat === 'text') {
      return trimmed;
    }

    // JSON mode: parse and validate
    try {
      const parsed = JSON.parse(trimmed);
      if (Array.isArray(parsed) || (typeof parsed === 'object' && parsed !== null)) {
        return parsed as UserPrompt;
      }
      if (typeof parsed === 'string') {
        return parsed;
      }
      setPromptError('JSON must be an object, array, or string');
      return null;
    } catch (e) {
      setPromptError(`Invalid JSON: ${(e as Error).message}`);
      return null;
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isStreaming || isAwaitingTools) return;

    setPromptError(null);
    const parsedPrompt = parsePrompt();
    if (parsedPrompt === null) return;

    sendMessage(parsedPrompt, { agent_type: agentType });
    setPrompt('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  };

  return (
    <div className="flex h-full flex-col bg-zinc-50 dark:bg-zinc-900">
      {/* Header */}
      <header className="flex items-center justify-between border-b border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950">
        <div>
          <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
            {title || 'New Chat'}
          </h1>
          {agentUuid && (
            <p className="text-xs text-zinc-500 dark:text-zinc-400">
              {agentUuid.slice(0, 8)}...
            </p>
          )}
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={startNewConversation}
          disabled={isStreaming || isAwaitingTools}
          className="gap-1"
        >
          <Plus className="h-4 w-4" />
          New Chat
        </Button>
      </header>

      {/* Chat Thread */}
      <ChatThread
        turns={turns}
        isLoadingHistory={isLoadingHistory}
        hasMore={hasMore}
        isAwaitingTools={isAwaitingTools}
        pendingFrontendTools={pendingFrontendTools}
        onLoadMore={loadMore}
        onToolResponse={handleToolResponse}
        className="flex-1"
      />

      {/* Error display */}
      {error && (
        <div className="mx-4 mb-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
          Error: {error}
        </div>
      )}

      {/* Input Form */}
      <footer className="border-t border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
        <form onSubmit={handleSubmit} className="mx-auto max-w-3xl space-y-2">
          {/* Controls row */}
          <div className="flex gap-2">
            <Select
              value={agentType}
              onValueChange={(value) => setAgentType(value as AgentType)}
              disabled={isStreaming || isAwaitingTools}
            >
              <SelectTrigger className="w-[180px]">
                <SelectValue placeholder="Select agent" />
              </SelectTrigger>
              <SelectContent>
                {AGENT_TYPES.map((type) => (
                  <SelectItem key={type.value} value={type.value}>
                    {type.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select
              value={promptFormat}
              onValueChange={(value) => {
                setPromptFormat(value as PromptFormat);
                setPromptError(null);
              }}
              disabled={isStreaming || isAwaitingTools}
            >
              <SelectTrigger className="w-[100px]">
                <SelectValue placeholder="Format" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="text">Text</SelectItem>
                <SelectItem value="json">JSON</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Textarea + submit row */}
          <div className="flex gap-2">
            <textarea
              value={prompt}
              onChange={(e) => {
                setPrompt(e.target.value);
                setPromptError(null);
              }}
              onKeyDown={handleKeyDown}
              placeholder={
                promptFormat === 'text'
                  ? 'Type a message...'
                  : 'Enter JSON array or object...'
              }
              disabled={isStreaming || isAwaitingTools}
              rows={2}
              className="flex-1 resize-none rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm shadow-xs outline-none transition-colors placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-400 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:placeholder:text-zinc-500 dark:focus:border-zinc-500 dark:focus:ring-zinc-500"
            />
            <Button
              type="submit"
              disabled={isStreaming || isAwaitingTools || !prompt.trim()}
              className="self-end"
            >
              <Send className="h-4 w-4" />
              <span className="ml-2 hidden sm:inline">Send</span>
            </Button>
          </div>

          {/* Error message */}
          {promptError && (
            <p className="text-sm text-red-600 dark:text-red-400">{promptError}</p>
          )}

          {/* Hint */}
          <p className="text-xs text-zinc-400 dark:text-zinc-500">
            {promptFormat === 'json'
              ? 'Paste a JSON object or array. Press Cmd/Ctrl+Enter to send.'
              : 'Press Cmd/Ctrl+Enter to send.'}
          </p>
        </form>
      </footer>
    </div>
  );
}
