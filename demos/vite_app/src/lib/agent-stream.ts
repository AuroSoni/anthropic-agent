import type { AgentNode } from './xml-parser';
import { AnthropicStreamParser } from './anthropic-parser';
import type { AnthropicEvent } from './anthropic-types';

export interface AgentState {
  status: 'idle' | 'streaming' | 'complete' | 'error';
  nodes: AgentNode[];
  metadata?: AgentMetadata;
  completion?: AgentCompletion;
  error?: string;
  rawContent: string; // Kept for debugging/inspection
}

export interface AgentMetadata {
  agent_uuid: string;
  model: string;
}

export interface AgentCompletion {
  total_steps: number;
  stop_reason: string;
  usage: {
    input_tokens: number;
    output_tokens: number;
  };
  container_id?: string;
}

export interface AgentConfig {
  system_prompt?: string;
  model?: string;
  max_steps?: number;
  // Add other config options as needed
}

export type AgentStreamCallbacks = {
  onUpdate: (state: AgentState) => void;
  onComplete?: (state: AgentState) => void;
  onError?: (error: string) => void;
};

export async function streamAgent(
  userPrompt: string,
  config: AgentConfig | undefined,
  callbacks: AgentStreamCallbacks
) {
  const initialState: AgentState = {
    status: 'streaming',
    nodes: [],
    rawContent: '',
  };
  
  // Notify start
  callbacks.onUpdate(initialState);

  let currentState = { ...initialState };
  const parser = new AnthropicStreamParser();

  try {
    const response = await fetch('http://localhost:8000/agent/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_prompt: userPrompt,
        agent_config: config,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('Response body is not readable');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      buffer += chunk;
      
      const lines = buffer.split('\n\n');
      // Keep the last part in the buffer if it's potentially incomplete
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataStr = line.slice(6);
          if (dataStr === '[DONE]') continue;

          try {
            const eventData = JSON.parse(dataStr);
            
            if (eventData.event === 'metadata') {
              currentState = {
                ...currentState,
                metadata: {
                  agent_uuid: eventData.agent_uuid,
                  model: eventData.model,
                },
              };
            } else if (eventData.event === 'chunk') {
              // Accumulate raw content for debugging
              currentState.rawContent += eventData.content;

              // Parse the content as an Anthropic Event
              try {
                const anthropicEvent = JSON.parse(eventData.content) as AnthropicEvent;
                parser.processEvent(anthropicEvent);
                
                // Update nodes from parser state
                currentState.nodes = parser.getNodes();
              } catch (e) {
                // Fallback: If it's not valid JSON or not a recognized event,
                // we might be receiving plain text or legacy XML fragments.
                // In a strict "raw" mode this shouldn't happen, but for robustness:
                console.warn('Failed to parse Anthropic event JSON:', e);
              }
              
              currentState = { ...currentState }; // Trigger update
            } else if (eventData.event === 'complete') {
              currentState = {
                ...currentState,
                status: 'complete',
                completion: {
                  total_steps: eventData.total_steps,
                  stop_reason: eventData.stop_reason,
                  usage: eventData.usage,
                  container_id: eventData.container_id,
                },
              };
            } else if (eventData.event === 'error') {
               currentState = {
                ...currentState,
                status: 'error',
                error: eventData.error,
              };
              if (callbacks.onError) callbacks.onError(eventData.error);
            }
            
            callbacks.onUpdate(currentState);

          } catch (e) {
            console.warn('Failed to parse SSE event:', dataStr, e);
          }
        }
      }
    }
    
    if (callbacks.onComplete) {
      callbacks.onComplete(currentState);
    }

  } catch (error: any) {
    currentState = {
      ...currentState,
      status: 'error',
      error: error.message,
    };
    callbacks.onUpdate(currentState);
    if (callbacks.onError) callbacks.onError(error.message);
  }
}
