import { useState, useCallback, useRef } from 'react';
import { streamAgent, type AgentState, type AgentConfig } from '../lib/agent-stream';

export function useAgent() {
  const [state, setState] = useState<AgentState>({
    status: 'idle',
    nodes: [],
    rawContent: '',
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const runAgent = useCallback(async (prompt: string, config?: AgentConfig) => {
    // Abort previous run if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    
    // Note: The current streamAgent implementation doesn't support AbortController yet,
    // but it's good practice to have the ref ready for future enhancements.
    // To properly support cancellation, we'd need to pass an AbortSignal to streamAgent.

    setState({
      status: 'streaming',
      nodes: [],
      rawContent: '',
    });

    await streamAgent(prompt, config, {
      onUpdate: (newState) => {
        setState(newState);
      },
      onError: (error) => {
        console.error('Agent stream error:', error);
      }
    });
  }, []);

  return {
    state,
    runAgent,
    isStreaming: state.status === 'streaming',
  };
}
