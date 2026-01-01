import { useState, type FormEvent } from 'react';
import { useAgent } from '@/hooks/use-agent';
import { NodeListRenderer } from './node-renderer';
import { StreamingIndicator } from './streaming-indicator';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
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
import { AGENT_TYPES, type AgentType } from '@/lib/agent-stream';

/**
 * AgentViewer is the main container for viewing agent output.
 * Includes an input form for prompts and renders the streamed node tree.
 */
export function AgentViewer() {
  const { state, runAgent, isStreaming } = useAgent();
  const [prompt, setPrompt] = useState('');
  const [agentType, setAgentType] = useState<AgentType>('agent_all_raw');
  const [copied, setCopied] = useState(false);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || isStreaming) return;
    
    runAgent(prompt.trim(), { agent_type: agentType });
    setPrompt('');
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
        <form onSubmit={handleSubmit} className="mx-auto flex max-w-3xl gap-2">
          <Select
            value={agentType}
            onValueChange={(value) => setAgentType(value as AgentType)}
            disabled={isStreaming}
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
          <Input
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter your prompt..."
            disabled={isStreaming}
            className="flex-1"
          />
          <Button type="submit" disabled={isStreaming || !prompt.trim()}>
            <Send className="h-4 w-4" />
            <span className="ml-2">Send</span>
          </Button>
        </form>
      </footer>
    </div>
  );
}

