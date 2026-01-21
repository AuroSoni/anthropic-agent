import { fetchEventSource } from '@microsoft/fetch-event-source';
import {
  type AgentNode,
  type AnthropicEvent,
  type StreamFormat,
  type PendingFrontendTool,
  type FrontendToolResult,
  AnthropicStreamParser,
  XmlStreamParser,
  parseMetaInit,
  stripMetaInit,
} from './parsers';

export interface AgentState {
  status: 'idle' | 'streaming' | 'complete' | 'error' | 'awaiting_tools';
  nodes: AgentNode[];
  metadata?: AgentMetadata;
  completion?: AgentCompletion;
  error?: string;
  rawContent: string; // Kept for debugging/inspection
  streamFormat?: StreamFormat; // Format detected from meta_init
  pendingFrontendTools?: PendingFrontendTool[]; // Frontend tools waiting for browser execution
}

export interface AgentMetadata {
  agent_uuid: string;
  model: string;
  user_query?: string;
  message_history?: unknown[];
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
export type AgentType = 'agent_no_tools' | 'agent_client_tools' | 'agent_all_raw' | 'agent_all_xml' | 'agent_frontend_tools';

/**
 * User prompt can be a simple string or a complex JSON structure (array or object)
 * matching the shapes used in the FastAPI backend (see agent_api_test.ipynb).
 */
export type UserPrompt = string | unknown[] | Record<string, unknown>;

export const AGENT_TYPES: { value: AgentType; label: string; description: string }[] = [
  { value: 'agent_no_tools', label: 'No Tools', description: 'Basic assistant without tools' },
  { value: 'agent_client_tools', label: 'Client Tools', description: 'With client-side tools only' },
  { value: 'agent_all_raw', label: 'All Tools (Raw)', description: 'All tools with raw JSON streaming' },
  { value: 'agent_all_xml', label: 'All Tools (XML)', description: 'All tools with XML streaming' },
  { value: 'agent_frontend_tools', label: 'Frontend Tools', description: 'With browser-executed tools (user_confirm)' },
];

export interface AgentConfig {
  system_prompt?: string;
  model?: string;
  max_steps?: number;
  formatter?: StreamFormat;
  agent_type?: AgentType;
  agent_uuid?: string; // UUID to resume existing conversation
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
  userPrompt: UserPrompt,
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
  let xmlFallbackParser: XmlStreamParser | null = null; // For XML chunks in raw mode (e.g., awaiting_frontend_tools)
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
      const trimmed = content.trim();
      
      // Check for XML content injected by backend (e.g., <awaiting_frontend_tools>)
      if (trimmed.startsWith('<')) {
        xmlFallbackParser?.appendChunk(content);
      } else {
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
      }
    } else {
      // XML format: content is plain XML text (unescaping happens in normalizeNodes)
      (parser as XmlStreamParser).appendChunk(content);
    }
    
    // Update nodes from parser state (aggressive real-time updates)
    // Merge main parser nodes with XML fallback nodes for raw mode
    const mainNodes = parser.getNodes();
    const xmlNodes = xmlFallbackParser?.getNodes() || [];
    currentState.nodes = [...mainNodes, ...xmlNodes];
    
    // Check for awaiting_frontend_tools tag (frontend tool relay)
    const awaitingNode = currentState.nodes.find(
      n => n.type === 'element' && n.tagName === 'awaiting_frontend_tools'
    );
    if (awaitingNode?.attributes?.data) {
      try {
        const toolsData = JSON.parse(awaitingNode.attributes.data) as PendingFrontendTool[];
        currentState.status = 'awaiting_tools';
        currentState.pendingFrontendTools = toolsData;
      } catch (e) {
        console.warn('Failed to parse awaiting_frontend_tools data:', e);
      }
    }
    
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
        agent_uuid: config?.agent_uuid,
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
              xmlFallbackParser = new XmlStreamParser(); // For XML chunks in raw mode
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
            xmlFallbackParser = new XmlStreamParser(); // For XML chunks in raw mode
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

  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    currentState = {
      ...currentState,
      status: 'error',
      error: errorMessage,
    };
    callbacks.onUpdate(currentState);
    if (callbacks.onError) callbacks.onError(errorMessage);
  }
}

/**
 * Submit frontend tool results and continue streaming the agent response.
 * Called after user interacts with frontend tools (e.g., confirms dialog).
 */
export async function streamToolResults(
  agentUuid: string,
  toolResults: FrontendToolResult[],
  callbacks: AgentStreamCallbacks,
  existingState: AgentState
) {
  // Continue from existing state, but switch back to streaming
  let currentState: AgentState = {
    ...existingState,
    status: 'streaming',
    pendingFrontendTools: undefined,
  };
  
  callbacks.onUpdate(currentState);
  
  // Parser for continuation stream - reuse format from existing state
  let parser: StreamParser | null = null;
  let xmlFallbackParser: XmlStreamParser | null = null; // For XML chunks in raw mode
  const streamFormat = existingState.streamFormat || 'xml';
  
  if (streamFormat === 'raw') {
    parser = new AnthropicStreamParser();
    xmlFallbackParser = new XmlStreamParser(); // For XML chunks in raw mode
  } else {
    parser = new XmlStreamParser();
  }

  /**
   * Process content for continuation stream.
   */
  function processContent(content: string) {
    if (!parser || !content.trim()) return;
    
    currentState.rawContent += content;

    if (streamFormat === 'raw') {
      const trimmed = content.trim();
      
      // Check for XML content injected by backend (e.g., <awaiting_frontend_tools>)
      if (trimmed.startsWith('<')) {
        xmlFallbackParser?.appendChunk(content);
      } else {
        const unescaped = content.replace(/\\\\/g, '\\');
        try {
          const anthropicEvent = JSON.parse(unescaped) as AnthropicEvent;
          (parser as AnthropicStreamParser).processEvent(anthropicEvent);
        } catch (e) {
          console.warn('Failed to parse Anthropic event JSON:', unescaped.slice(0, 100), e);
        }
      }
    } else {
      (parser as XmlStreamParser).appendChunk(content);
    }
    
    // Merge new nodes with existing nodes
    // Include XML fallback nodes for raw mode
    const mainNodes = parser.getNodes();
    const xmlNodes = xmlFallbackParser?.getNodes() || [];
    const newNodes = [...mainNodes, ...xmlNodes];
    currentState.nodes = [...existingState.nodes, ...newNodes];
    
    // Check for another awaiting_frontend_tools (nested tool calls)
    const awaitingNode = newNodes.find(
      (n: AgentNode) => n.type === 'element' && n.tagName === 'awaiting_frontend_tools'
    );
    if (awaitingNode?.attributes?.data) {
      try {
        const toolsData = JSON.parse(awaitingNode.attributes.data) as PendingFrontendTool[];
        currentState.status = 'awaiting_tools';
        currentState.pendingFrontendTools = toolsData;
      } catch (e) {
        console.warn('Failed to parse awaiting_frontend_tools data:', e);
      }
    }
    
    currentState = { ...currentState };
    callbacks.onUpdate(currentState);
  }

  try {
    await fetchEventSource('http://localhost:8000/agent/tool_results', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        agent_uuid: agentUuid,
        tool_results: toolResults,
      }),

      onopen: async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      },

      onmessage: (ev) => {
        const dataStr = ev.data;
        
        if (dataStr === '[DONE]') {
          if (currentState.status === 'streaming') {
            currentState = { ...currentState, status: 'complete' };
            callbacks.onUpdate(currentState);
          }
          return;
        }

        processContent(dataStr);
      },

      onclose: () => {
        if (currentState.status === 'streaming') {
          currentState = { ...currentState, status: 'complete' };
          callbacks.onUpdate(currentState);
        }
        if (callbacks.onComplete) {
          callbacks.onComplete(currentState);
        }
      },

      onerror: (err) => {
        throw err;
      },
    });

  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    currentState = {
      ...currentState,
      status: 'error',
      error: errorMessage,
    };
    callbacks.onUpdate(currentState);
    if (callbacks.onError) callbacks.onError(errorMessage);
  }
}

// Re-export types for convenience
export type { PendingFrontendTool, FrontendToolResult };
