import { useState, type FormEvent, type KeyboardEvent } from 'react';
import { useAgent } from '@/hooks/use-agent';
import { NodeListRenderer } from './node-renderer';
import { StreamingIndicator } from './streaming-indicator';
import { FrontendToolPrompt } from './frontend-tool-prompt';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Send, Copy, Check } from 'lucide-react';
import { AGENT_TYPES, type AgentType, type UserPrompt } from '@/lib/agent-stream';
import { createToolResult } from '@/lib/frontend-tools';
import type { PendingFrontendTool } from '@/lib/parsers/types';

type PromptFormat = 'text' | 'json';

/**
 * AgentViewer is the main container for viewing agent output.
 * Includes an input form for prompts and renders the streamed node tree.
 */
export function AgentViewer() {
  const { state, runAgent, submitToolResults, isStreaming, isAwaitingTools } = useAgent();
  const [prompt, setPrompt] = useState('');
  const [agentType, setAgentType] = useState<AgentType>('agent_all_xml');
  const [promptFormat, setPromptFormat] = useState<PromptFormat>('text');
  const [promptError, setPromptError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  /**
   * Handle frontend tool response from user interaction.
   * Creates tool results for all pending tools and submits them to continue the agent.
   */
  const handleToolResponse = (respondedTool: PendingFrontendTool, response: string) => {
    // Create results for all pending tools
    // For the responded tool, use the actual response; others get a placeholder
    const results = state.pendingFrontendTools?.map(tool => 
      tool.tool_use_id === respondedTool.tool_use_id 
        ? createToolResult(tool, response)
        : createToolResult(tool, 'skipped')
    ) ?? [];
    
    // If there's only one pending tool, just submit that result
    if (state.pendingFrontendTools?.length === 1) {
      submitToolResults([createToolResult(respondedTool, response)]);
    } else {
      // Multiple tools: submit all results
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
      // Must be array or object (not null, not primitive)
      if (Array.isArray(parsed) || (typeof parsed === 'object' && parsed !== null)) {
        return parsed as UserPrompt;
      }
      // Allow string too (valid JSON string)
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

    runAgent(parsedPrompt, { agent_type: agentType });
    setPrompt('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit on Cmd/Ctrl+Enter
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSubmit(e as unknown as FormEvent);
    }
  };

  const handleCopy = async () => {
    if (state.nodes.length === 0) return;
    
    try {
      const jsonContent = JSON.stringify(state.nodes, null, 2);
      await navigator.clipboard.writeText(jsonContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="flex h-screen flex-col bg-zinc-50 dark:bg-zinc-900">
      {/* Header */}
      <header className="border-b border-zinc-200 bg-white px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950">
        <h1 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
          Agent Viewer
        </h1>
        {state.metadata && (
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            Model: {state.metadata.model} | Session: {state.metadata.agent_uuid?.slice(0, 8)}...
          </p>
        )}
      </header>

      {/* Content Area */}
      <ScrollArea className="flex-1 p-4">
        <Card className="mx-auto max-w-3xl p-4">
          {state.nodes.length === 0 && state.status === 'idle' && (
            <div className="py-8 text-center text-zinc-500 dark:text-zinc-400">
              Enter a prompt to start the agent.
            </div>
          )}

          {state.nodes.length > 0 && (
            <div>
              <NodeListRenderer nodes={state.nodes} />
            </div>
          )}

          {isStreaming && <StreamingIndicator />}

          {/* Frontend tool prompts - shown when agent is awaiting tool execution */}
          {isAwaitingTools && state.pendingFrontendTools && state.pendingFrontendTools.length > 0 && (
            <div className="mt-4">
              {state.pendingFrontendTools.map(tool => (
                <FrontendToolPrompt
                  key={tool.tool_use_id}
                  tool={tool}
                  onSubmit={(response) => handleToolResponse(tool, response)}
                />
              ))}
            </div>
          )}

          {state.status === 'error' && state.error && (
            <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300">
              Error: {state.error}
            </div>
          )}

          {state.status === 'complete' && state.completion && (
            <div className="mt-4 border-t border-zinc-200 pt-3 text-xs text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
              Completed in {state.completion.total_steps} step(s) | 
              Tokens: {state.completion.usage.input_tokens} in / {state.completion.usage.output_tokens} out
            </div>
          )}

          {/* Copy button - shown when complete and has content */}
          {state.status === 'complete' && state.nodes.length > 0 && (
            <div className="mt-3 flex justify-end">
              <Button
                variant="ghost"
                size="icon"
                onClick={handleCopy}
                className="h-7 w-7 text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
                title="Copy response"
              >
                {copied ? (
                  <Check className="h-4 w-4 text-emerald-500" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>
          )}
        </Card>
      </ScrollArea>

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
                  ? 'Enter your prompt...'
                  : 'Enter JSON array or object...'
              }
              disabled={isStreaming || isAwaitingTools}
              rows={3}
              className="flex-1 resize-none rounded-md border border-zinc-200 bg-white px-3 py-2 text-sm shadow-xs outline-none transition-colors placeholder:text-zinc-400 focus:border-zinc-400 focus:ring-1 focus:ring-zinc-400 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100 dark:placeholder:text-zinc-500 dark:focus:border-zinc-500 dark:focus:ring-zinc-500"
            />
            <Button
              type="submit"
              disabled={isStreaming || isAwaitingTools || !prompt.trim()}
              className="self-end"
            >
              <Send className="h-4 w-4" />
              <span className="ml-2">Send</span>
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

