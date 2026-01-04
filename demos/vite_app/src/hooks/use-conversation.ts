import { useState, useCallback, useEffect, useRef } from 'react';
import {
  streamAgent,
  streamToolResults,
  type AgentState,
  type AgentConfig,
  type UserPrompt,
  type FrontendToolResult,
} from '../lib/agent-stream';
import { fetchConversationHistory, generateConversationTitle } from '../lib/api';
import { useAgentUrlState } from './use-url-state';
import { convertMessagesToNodes } from '../lib/message-converter';
import type { AgentNode } from '../lib/parsers/types';

/**
 * A conversation turn representing a user prompt and agent response pair.
 */
export interface ConversationTurn {
  id: string; // conversation_id or temporary id for streaming
  sequenceNumber: number;
  userMessage: string;
  assistantResponse: string | null;
  messages: unknown[]; // Full message array from API
  nodes?: AgentNode[]; // Parsed nodes for rich rendering (from history or streaming)
  isStreaming: boolean;
  streamingState?: AgentState; // Live streaming state for current turn
  completedAt?: string;
  usage?: {
    input_tokens: number;
    output_tokens: number;
  };
  totalSteps?: number;
}

export interface ConversationState {
  turns: ConversationTurn[];
  hasMore: boolean;
  isLoadingHistory: boolean;
  isStreaming: boolean;
  isAwaitingTools: boolean;
  error?: string;
  pendingFrontendTools?: AgentState['pendingFrontendTools'];
  title?: string;
  isGeneratingTitle: boolean;
}

/**
 * Hook for managing a full conversation thread with history loading and streaming.
 */
export function useConversation() {
  const { agentUuid, setAgentUuid, hasAgent } = useAgentUrlState();
  
  const [state, setState] = useState<ConversationState>({
    turns: [],
    hasMore: false,
    isLoadingHistory: false,
    isStreaming: false,
    isAwaitingTools: false,
    isGeneratingTitle: false,
  });

  // Track the current streaming state for the active turn
  const currentStreamingStateRef = useRef<AgentState | null>(null);
  
  // Track the smallest sequence number for pagination cursor
  const oldestSequenceRef = useRef<number | null>(null);

  /**
   * Load conversation history from the API.
   * Called on mount if agentUuid exists, and on scroll-up for older messages.
   */
  const loadHistory = useCallback(async (before?: number) => {
    if (!agentUuid) return;
    
    setState(prev => ({ ...prev, isLoadingHistory: true, error: undefined }));
    
    try {
      const response = await fetchConversationHistory(agentUuid, before, 20);
      
      // Convert API response to ConversationTurns with parsed nodes for rich rendering
      const newTurns: ConversationTurn[] = response.conversations.map(conv => ({
        id: conv.conversation_id,
        sequenceNumber: conv.sequence_number,
        userMessage: conv.user_message,
        assistantResponse: conv.final_response,
        messages: conv.messages,
        nodes: conv.messages.length > 0 ? convertMessagesToNodes(conv.messages) : undefined,
        isStreaming: false,
        completedAt: conv.completed_at ?? undefined,
        usage: conv.usage ?? undefined,
        totalSteps: conv.total_steps ?? undefined,
      }));
      
      // Update oldest sequence for pagination
      if (newTurns.length > 0) {
        const minSeq = Math.min(...newTurns.map(t => t.sequenceNumber));
        if (oldestSequenceRef.current === null || minSeq < oldestSequenceRef.current) {
          oldestSequenceRef.current = minSeq;
        }
      }
      
      // API returns newest-first, but chat UI needs oldest-at-top, newest-at-bottom
      // Reverse for display order
      const turnsInDisplayOrder = [...newTurns].reverse();
      
      setState(prev => ({
        ...prev,
        // For initial load: use reversed turns (oldest first)
        // For "load more": prepend older turns before existing ones
        turns: before !== undefined 
          ? [...turnsInDisplayOrder, ...prev.turns]
          : turnsInDisplayOrder,
        hasMore: response.has_more,
        isLoadingHistory: false,
        // Set title from response (only on initial load)
        title: before === undefined ? (response.title ?? prev.title) : prev.title,
      }));
    } catch (error) {
      console.error('Failed to load conversation history:', error);
      setState(prev => ({
        ...prev,
        isLoadingHistory: false,
        error: error instanceof Error ? error.message : 'Failed to load history',
      }));
    }
  }, [agentUuid]);

  /**
   * Load more (older) conversations for infinite scroll.
   */
  const loadMore = useCallback(() => {
    if (state.isLoadingHistory || !state.hasMore || oldestSequenceRef.current === null) {
      return;
    }
    loadHistory(oldestSequenceRef.current);
  }, [state.isLoadingHistory, state.hasMore, loadHistory]);

  /**
   * Send a new message and stream the agent response.
   */
  const sendMessage = useCallback(async (prompt: UserPrompt, config?: AgentConfig) => {
    // Create a temporary turn for the streaming response
    const tempId = `temp-${Date.now()}`;
    const userMessageText = typeof prompt === 'string' 
      ? prompt 
      : JSON.stringify(prompt);
    
    const newTurn: ConversationTurn = {
      id: tempId,
      sequenceNumber: state.turns.length > 0 
        ? Math.max(...state.turns.map(t => t.sequenceNumber)) + 1 
        : 1,
      userMessage: userMessageText,
      assistantResponse: null,
      messages: [],
      isStreaming: true,
    };
    
    setState(prev => ({
      ...prev,
      turns: [...prev.turns, newTurn],
      isStreaming: true,
      isAwaitingTools: false,
      error: undefined,
    }));

    // Send both agent_uuid AND agent_type when resuming
    // agent_type ensures correct config (tools, db backend) is used
    const streamConfig = agentUuid 
      ? { agent_uuid: agentUuid, agent_type: config?.agent_type }
      : { agent_type: config?.agent_type };
    
    await streamAgent(prompt, streamConfig, {
      onUpdate: (streamState) => {
        currentStreamingStateRef.current = streamState;
        
        // Update the agent UUID from metadata if this is a new conversation
        if (streamState.metadata?.agent_uuid && !agentUuid) {
          setAgentUuid(streamState.metadata.agent_uuid);
          
          // Generate title for new conversation (parallel to streaming)
          setState(prev => ({ ...prev, isGeneratingTitle: true }));
          generateConversationTitle(
            streamState.metadata.agent_uuid,
            userMessageText,
            config?.agent_type ?? 'agent_frontend_tools'
          )
            .then(title => {
              setState(prev => ({ ...prev, title, isGeneratingTitle: false }));
            })
            .catch(error => {
              console.error('Failed to generate title:', error);
              setState(prev => ({ ...prev, isGeneratingTitle: false }));
            });
        }
        
        // Update the streaming turn with live state
        setState(prev => {
          const updatedTurns = prev.turns.map(turn =>
            turn.id === tempId
              ? {
                  ...turn,
                  streamingState: streamState,
                  isStreaming: streamState.status === 'streaming',
                }
              : turn
          );
          
          return {
            ...prev,
            turns: updatedTurns,
            isStreaming: streamState.status === 'streaming',
            isAwaitingTools: streamState.status === 'awaiting_tools',
            pendingFrontendTools: streamState.pendingFrontendTools,
            error: streamState.status === 'error' ? streamState.error : undefined,
          };
        });
        
        // When complete, finalize the turn
        if (streamState.status === 'complete' || streamState.status === 'error') {
          setState(prev => {
            const updatedTurns = prev.turns.map(turn =>
              turn.id === tempId
                ? {
                    ...turn,
                    isStreaming: false,
                    streamingState: undefined, // Clear to prevent stale state matching
                    nodes: streamState.nodes, // Preserve nodes for rich rendering
                    assistantResponse: extractFinalResponse(streamState),
                    usage: streamState.completion?.usage,
                    totalSteps: streamState.completion?.total_steps,
                  }
                : turn
            );
            
            return {
              ...prev,
              turns: updatedTurns,
              isStreaming: false,
            };
          });
        }
      },
      onError: (error) => {
        console.error('Stream error:', error);
        setState(prev => ({
          ...prev,
          isStreaming: false,
          error,
        }));
      },
    });
  }, [state.turns, agentUuid, setAgentUuid]);

  /**
   * Submit frontend tool results to continue the agent.
   */
  const submitToolResults = useCallback(async (results: FrontendToolResult[]) => {
    if (!agentUuid || !currentStreamingStateRef.current) {
      console.error('Cannot submit tool results: missing agent state');
      return;
    }
    
    setState(prev => ({
      ...prev,
      isStreaming: true,
      isAwaitingTools: false,
      pendingFrontendTools: undefined,
    }));
    
    await streamToolResults(
      agentUuid,
      results,
      {
        onUpdate: (streamState) => {
          currentStreamingStateRef.current = streamState;
          
          // Update the current streaming turn (use findLastIndex to get the most recent turn)
          setState(prev => {
            const currentTurnIndex = prev.turns.findLastIndex(t => t.isStreaming || t.streamingState);
            if (currentTurnIndex === -1) return prev;
            
            const updatedTurns = [...prev.turns];
            updatedTurns[currentTurnIndex] = {
              ...updatedTurns[currentTurnIndex],
              streamingState: streamState,
              isStreaming: streamState.status === 'streaming',
            };
            
            return {
              ...prev,
              turns: updatedTurns,
              isStreaming: streamState.status === 'streaming',
              isAwaitingTools: streamState.status === 'awaiting_tools',
              pendingFrontendTools: streamState.pendingFrontendTools,
            };
          });
          
          if (streamState.status === 'complete' || streamState.status === 'error') {
            setState(prev => {
              const currentTurnIndex = prev.turns.findLastIndex(t => t.streamingState);
              if (currentTurnIndex === -1) return prev;
              
              const updatedTurns = [...prev.turns];
              updatedTurns[currentTurnIndex] = {
                ...updatedTurns[currentTurnIndex],
                isStreaming: false,
                streamingState: undefined, // Clear to prevent stale state matching
                nodes: streamState.nodes, // Preserve nodes for rich rendering
                assistantResponse: extractFinalResponse(streamState),
                usage: streamState.completion?.usage,
                totalSteps: streamState.completion?.total_steps,
              };
              
              return {
                ...prev,
                turns: updatedTurns,
                isStreaming: false,
              };
            });
          }
        },
        onError: (error) => {
          console.error('Tool results stream error:', error);
          setState(prev => ({
            ...prev,
            isStreaming: false,
            error,
          }));
        },
      },
      currentStreamingStateRef.current
    );
  }, [agentUuid]);

  /**
   * Start a new conversation (clears history and URL).
   */
  const startNewConversation = useCallback(() => {
    setAgentUuid(null);
    oldestSequenceRef.current = null;
    currentStreamingStateRef.current = null;
    setState({
      turns: [],
      hasMore: false,
      isLoadingHistory: false,
      isStreaming: false,
      isAwaitingTools: false,
      title: undefined,
      isGeneratingTitle: false,
    });
  }, [setAgentUuid]);

  // Load history on mount if agent_uuid exists in URL
  useEffect(() => {
    if (hasAgent && state.turns.length === 0 && !state.isLoadingHistory) {
      loadHistory();
    }
  }, [hasAgent, state.turns.length, state.isLoadingHistory, loadHistory]);

  return {
    ...state,
    agentUuid,
    sendMessage,
    submitToolResults,
    loadMore,
    startNewConversation,
  };
}

/**
 * Extract a plain text final response from the streaming state.
 */
function extractFinalResponse(state: AgentState): string | null {
  if (!state.nodes || state.nodes.length === 0) return null;
  
  // Find text nodes that aren't inside thinking blocks
  const textParts: string[] = [];
  
  for (const node of state.nodes) {
    if (node.type === 'text' && node.content) {
      textParts.push(node.content);
    } else if (node.type === 'element' && node.tagName === 'text' && node.children) {
      for (const child of node.children) {
        if (child.type === 'text' && child.content) {
          textParts.push(child.content);
        }
      }
    }
  }
  
  return textParts.length > 0 ? textParts.join('') : null;
}

