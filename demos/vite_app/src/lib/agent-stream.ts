import {
  type AgentNode,
  type AnthropicEvent,
  type StreamFormat,
  AnthropicStreamParser,
  XmlStreamParser,
  parseMetaInit,
  stripMetaInit,
} from './parsers';

export interface AgentState {
  status: 'idle' | 'streaming' | 'complete' | 'error';
  nodes: AgentNode[];
  metadata?: AgentMetadata;
  completion?: AgentCompletion;
  error?: string;
  rawContent: string; // Kept for debugging/inspection
  streamFormat?: StreamFormat; // Format detected from meta_init
}

export interface AgentMetadata {
  agent_uuid: string;
  model: string;
  user_query?: string;
  message_history?: any[];
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
  formatter?: StreamFormat;
  // Add other config options as needed
}

export type AgentStreamCallbacks = {
  onUpdate: (state: AgentState) => void;
  onComplete?: (state: AgentState) => void;
  onError?: (error: string) => void;
};

/**
 * Union type for both parser implementations.
 */
type StreamParser = AnthropicStreamParser | XmlStreamParser;

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
  
  // Parser selection: will be determined from meta_init
  let parser: StreamParser | null = null;
  let streamFormat: StreamFormat | null = null;
  let metaInitProcessed = false;

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
          if (dataStr === '[DONE]') {
            // Stream complete - mark as complete if not already
            if (currentState.status === 'streaming') {
              currentState = {
                ...currentState,
                status: 'complete',
              };
              callbacks.onUpdate(currentState);
            }
            continue;
          }

          // Check for meta_init on first data chunk
          if (!metaInitProcessed) {
            const metaInit = parseMetaInit(dataStr);
            if (metaInit) {
              streamFormat = metaInit.format;
              currentState.streamFormat = streamFormat;
              
              // Update metadata from meta_init
              currentState.metadata = {
                ...currentState.metadata,
                agent_uuid: metaInit.agent_uuid,
                model: metaInit.model,
                user_query: metaInit.user_query,
                message_history: metaInit.message_history,
              };
              
              // Initialize parser based on format
              if (streamFormat === 'raw') {
                parser = new AnthropicStreamParser();
              } else {
                parser = new XmlStreamParser();
              }
              
              metaInitProcessed = true;
              
              // Strip meta_init and continue with remaining content
              const remaining = stripMetaInit(dataStr);
              if (!remaining.trim()) {
                callbacks.onUpdate(currentState);
                continue;
              }
              
              // Process remaining content after meta_init
              processContent(remaining);
              continue;
            }
            
            // No meta_init found - detect format from content
            // If it starts with '{', assume raw JSON; otherwise XML
            if (dataStr.trim().startsWith('{')) {
              parser = new AnthropicStreamParser();
              streamFormat = 'raw';
            } else {
              parser = new XmlStreamParser();
              streamFormat = 'xml';
            }
            currentState.streamFormat = streamFormat;
            metaInitProcessed = true;
          }
          
          processContent(dataStr);
        }
      }
    }
    
    // Ensure complete state is set
    if (currentState.status === 'streaming') {
      currentState = { ...currentState, status: 'complete' };
      callbacks.onUpdate(currentState);
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

  /**
   * Process content based on detected stream format.
   */
  function processContent(content: string) {
    if (!parser || !content.trim()) return;
    
    // Accumulate raw content for debugging
    currentState.rawContent += content;

    // Process content based on format
    if (streamFormat === 'raw') {
      // Raw format: content is JSON-encoded Anthropic events
      try {
        const anthropicEvent = JSON.parse(content) as AnthropicEvent;
        (parser as AnthropicStreamParser).processEvent(anthropicEvent);
      } catch (e) {
        // Not valid JSON - might be an error message or other text
        console.warn('Failed to parse Anthropic event JSON:', content.slice(0, 100), e);
      }
    } else {
      // XML format: content is plain XML text
      (parser as XmlStreamParser).appendChunk(content);
    }
    
    // Update nodes from parser state (aggressive real-time updates)
    currentState.nodes = parser.getNodes();
    currentState = { ...currentState }; // Trigger update
    callbacks.onUpdate(currentState);
  }
}
