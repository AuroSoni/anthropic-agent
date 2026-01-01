import { fetchEventSource } from '@microsoft/fetch-event-source';
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

/**
 * Available agent types matching the backend configuration.
 */
export type AgentType = 'agent_no_tools' | 'agent_client_tools' | 'agent_all_raw' | 'agent_all_xml';

export const AGENT_TYPES: { value: AgentType; label: string; description: string }[] = [
  { value: 'agent_no_tools', label: 'No Tools', description: 'Basic assistant without tools' },
  { value: 'agent_client_tools', label: 'Client Tools', description: 'With client-side tools only' },
  { value: 'agent_all_raw', label: 'All Tools (Raw)', description: 'All tools with raw JSON streaming' },
  { value: 'agent_all_xml', label: 'All Tools (XML)', description: 'All tools with XML streaming' },
];

export interface AgentConfig {
  system_prompt?: string;
  model?: string;
  max_steps?: number;
  formatter?: StreamFormat;
  agent_type?: AgentType;
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
      // Backend escapes backslashes for SSE transport, so unescape before parsing
      const unescaped = content.replace(/\\\\/g, '\\');
      try {
        const anthropicEvent = JSON.parse(unescaped) as AnthropicEvent;
        (parser as AnthropicStreamParser).processEvent(anthropicEvent);
      } catch (e) {
        // Not valid JSON - might be an error message or other text
        console.warn('Failed to parse Anthropic event JSON:', unescaped.slice(0, 100), e);
      }
    } else {
      // XML format: content is plain XML text (unescaping happens in normalizeNodes)
      (parser as XmlStreamParser).appendChunk(content);
    }
    
    // Update nodes from parser state (aggressive real-time updates)
    currentState.nodes = parser.getNodes();
    currentState = { ...currentState }; // Trigger update
    callbacks.onUpdate(currentState);
  }

  try {
    await fetchEventSource('http://localhost:8000/agent/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_prompt: userPrompt,
        agent_type: config?.agent_type,
      }),

      onopen: async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      },

      onmessage: (ev) => {
        // DO NOT blanket unescape here - parsers handle escaping at the right level:
        // - XML parser: unescapes \n in text nodes (not attributes, so JSON.parse works)
        // - Raw parser: JSON.parse handles \n natively
        const dataStr = ev.data;
        
        if (dataStr === '[DONE]') {
          // Stream complete - mark as complete if not already
          if (currentState.status === 'streaming') {
            currentState = { ...currentState, status: 'complete' };
            callbacks.onUpdate(currentState);
          }
          return;
        }

        // Check for meta_init on first data chunk
        if (!metaInitProcessed) {
          // Don't unescape before parseMetaInit - the JSON uses standard escaping
          // that JSON.parse() handles correctly. Unescaping would break newlines in strings.
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
            
            // Strip meta_init and continue with remaining content (use raw dataStr)
            const remaining = stripMetaInit(dataStr);
            if (!remaining.trim()) {
              callbacks.onUpdate(currentState);
              return;
            }
            
            // Process remaining content after meta_init
            processContent(remaining);
            return;
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
      },

      onclose: () => {
        // Ensure complete state is set when stream closes
        if (currentState.status === 'streaming') {
          currentState = { ...currentState, status: 'complete' };
          callbacks.onUpdate(currentState);
        }
        if (callbacks.onComplete) {
          callbacks.onComplete(currentState);
        }
      },

      onerror: (err) => {
        // Rethrow to stop retries and let the catch block handle it
        throw err;
      },
    });

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
