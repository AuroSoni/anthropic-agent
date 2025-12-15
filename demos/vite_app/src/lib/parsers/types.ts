/**
 * Core types for the agent stream parser library.
 * These types are used across all parsers and can be exported for use in other projects.
 */

// ============================================================================
// Node Types (for parsed output tree)
// ============================================================================

export type AgentNodeType = 'text' | 'element';

export interface AgentNode {
  type: AgentNodeType;
  content?: string; // For text nodes
  tagName?: string; // For element nodes
  attributes?: Record<string, string>; // For element nodes
  children?: AgentNode[]; // For element nodes
}

// ============================================================================
// Anthropic Event Types (for raw format parsing)
// ============================================================================

export type AnthropicEventType = 
  | 'message_start'
  | 'message_delta'
  | 'message_stop'
  | 'content_block_start'
  | 'content_block_delta'
  | 'content_block_stop'
  | 'ping'
  | 'error';

export interface MessageStartEvent {
  type: 'message_start';
  message: {
    id: string;
    type: 'message';
    role: 'user' | 'assistant';
    model: string;
    content: any[];
    stop_reason: string | null;
    stop_sequence: string | null;
    usage: {
      input_tokens: number;
      output_tokens: number;
    };
  };
}

export interface MessageDeltaEvent {
  type: 'message_delta';
  delta: {
    stop_reason?: string;
    stop_sequence?: string;
  };
  usage?: {
    output_tokens: number;
  };
}

export interface MessageStopEvent {
  type: 'message_stop';
}

// Allow string for flexibility with new tool result types
export type ContentBlockType = 
  | 'text' 
  | 'thinking' 
  | 'tool_use' 
  | 'server_tool_use' 
  | 'redacted_thinking'
  | string; 

export interface ContentBlockStartEvent {
  type: 'content_block_start';
  index: number;
  content_block: {
    type: ContentBlockType;
    text?: string;
    thinking?: string;
    id?: string;
    name?: string;
    input?: any;
    content?: any; // For tool results
    tool_use_id?: string; // For tool results
    [key: string]: any;
  };
}

export interface ContentBlockDeltaEvent {
  type: 'content_block_delta';
  index: number;
  delta: {
    type: 'text_delta' | 'thinking_delta' | 'signature_delta' | 'input_json_delta';
    text?: string;
    thinking?: string;
    signature?: string;
    partial_json?: string;
    [key: string]: any;
  };
}

export interface ContentBlockStopEvent {
  type: 'content_block_stop';
  index: number;
}

export type AnthropicEvent = 
  | MessageStartEvent 
  | MessageDeltaEvent 
  | MessageStopEvent 
  | ContentBlockStartEvent 
  | ContentBlockDeltaEvent 
  | ContentBlockStopEvent;

// ============================================================================
// Stream Format Types
// ============================================================================

/**
 * Stream format type: "xml" or "raw"
 */
export type StreamFormat = 'xml' | 'raw';

/**
 * MetaInit contains initialization metadata sent at the start of a stream.
 * Emitted as <meta_init data="..."></meta_init> by the backend.
 */
export interface MetaInit {
  format: StreamFormat;
  user_query: string;
  message_history: any[];
  agent_uuid: string;
  model: string;
}

