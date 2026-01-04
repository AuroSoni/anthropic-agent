import { useState, useCallback, useRef } from 'react';
import {
  streamAgent,
  streamToolResults,
  type AgentState,
  type AgentConfig,
  type UserPrompt,
  type FrontendToolResult,
} from '../lib/agent-stream';

export function useAgent() {
  const [state, setState] = useState<AgentState>({
    status: 'idle',
    nodes: [],
    rawContent: '',
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const runAgent = useCallback(async (prompt: UserPrompt, config?: AgentConfig) => {
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

  /**
   * Submit frontend tool results and continue the agent stream.
   * Called after user interacts with frontend tools (e.g., clicks Yes/No on confirm dialog).
   */
  const submitToolResults = useCallback(async (results: FrontendToolResult[]) => {
    if (!state.metadata?.agent_uuid) {
      console.error('Cannot submit tool results: no agent_uuid in state');
      return;
    }
    
    // Transition to streaming state, clear pending tools
    setState(prev => ({
      ...prev,
      status: 'streaming',
      pendingFrontendTools: undefined,
    }));
    
    await streamToolResults(
      state.metadata.agent_uuid,
      results,
      {
        onUpdate: (newState) => {
          setState(newState);
        },
        onError: (error) => {
          console.error('Agent continuation error:', error);
        }
      },
      state
    );
  }, [state]);

  return {
    state,
    runAgent,
    submitToolResults,
    isStreaming: state.status === 'streaming',
    isAwaitingTools: state.status === 'awaiting_tools',
  };
}
