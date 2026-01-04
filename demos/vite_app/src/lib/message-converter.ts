/**
 * Convert Anthropic API message format to AgentNode[] for rendering.
 * This allows historical conversations to display the same rich formatting
 * as live streaming (thinking blocks, tool calls, tool results, etc.).
 */

import type { AgentNode } from './parsers/types';

/**
 * Anthropic content block types in saved messages.
 */
interface TextContent {
  type: 'text';
  text: string;
}

interface ThinkingContent {
  type: 'thinking';
  thinking: string;
  signature?: string;
}

interface ToolUseContent {
  type: 'tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

interface ToolResultContent {
  type: 'tool_result';
  tool_use_id: string;
  content: string | Array<{ type: string; text?: string }>;
  is_error?: boolean;
}

interface ServerToolUseContent {
  type: 'server_tool_use';
  id: string;
  name: string;
  input: Record<string, unknown>;
}

/**
 * Server tool results have dynamic type names like:
 * - bash_code_execution_tool_result
 * - web_search_tool_result
 * - text_editor_code_execution_tool_result
 * 
 * Note: Not included in ContentBlock union since type is dynamic.
 * Handled in default case of contentBlockToNode with type assertion.
 */
interface ServerToolResultContent {
  type: string; // Ends with '_tool_result'
  tool_use_id: string;
  content: unknown; // Can be string, array, or complex object
}

/**
 * Known content block types. ServerToolResultContent is handled
 * separately in the default case since its type is dynamic.
 */
type ContentBlock = 
  | TextContent 
  | ThinkingContent 
  | ToolUseContent 
  | ToolResultContent 
  | ServerToolUseContent;

interface Message {
  role: 'user' | 'assistant';
  content: ContentBlock[] | string;
}

/**
 * Track tool names by their IDs so tool_result can reference them.
 */
type ToolNameMap = Map<string, string>;

/**
 * Extract content from server tool results which can have complex structures.
 */
function extractServerToolResultContent(content: unknown): string {
  if (content === null || content === undefined) {
    return '';
  }
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content)) {
    // Handle array of content blocks (common in code execution results)
    return content
      .map(item => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object') {
          // Handle text blocks
          if ('text' in item && typeof item.text === 'string') return item.text;
          // Handle other structured content
          return JSON.stringify(item, null, 2);
        }
        return String(item);
      })
      .join('\n');
  }
  // Object content - serialize to JSON
  return JSON.stringify(content, null, 2);
}

/**
 * Convert a single content block to an AgentNode.
 */
function contentBlockToNode(block: ContentBlock, toolNames: ToolNameMap): AgentNode | null {
  switch (block.type) {
    case 'text':
      return {
        type: 'text',
        content: block.text,
      };

    case 'thinking':
      return {
        type: 'element',
        tagName: 'thinking',
        children: [{ type: 'text', content: block.thinking }],
      };

    case 'tool_use':
      // Store tool name for later tool_result lookup
      toolNames.set(block.id, block.name);
      
      return {
        type: 'element',
        tagName: 'tool_call',
        attributes: {
          name: block.name,
          id: block.id,
        },
        children: [
          {
            type: 'text',
            content: JSON.stringify(block.input, null, 2),
          },
        ],
      };

    case 'tool_result': {
      // Extract result content
      let resultContent: string;
      if (typeof block.content === 'string') {
        resultContent = block.content;
      } else if (Array.isArray(block.content)) {
        resultContent = block.content
          .filter((c): c is { type: string; text: string } => c.type === 'text' && !!c.text)
          .map(c => c.text)
          .join('\n');
      } else {
        resultContent = String(block.content);
      }

      const toolName = toolNames.get(block.tool_use_id);
      
      return {
        type: 'element',
        tagName: 'tool_result',
        attributes: {
          name: toolName || 'Unknown',
          is_error: block.is_error ? 'true' : 'false',
        },
        children: [{ type: 'text', content: resultContent }],
      };
    }

    case 'server_tool_use':
      // Server-side tool call (web_search, code_execution, etc.)
      toolNames.set(block.id, block.name);
      
      return {
        type: 'element',
        tagName: 'server_tool_call',  // Matches streaming parser output
        attributes: {
          name: block.name,
          id: block.id,
        },
        children: [
          {
            type: 'text',
            content: JSON.stringify(block.input, null, 2),
          },
        ],
      };

    default: {
      // Handle server tool results (*_tool_result variants like bash_code_execution_tool_result)
      // Cast to unknown first to access dynamic type property
      const unknownBlock = block as unknown as { type: string };
      if (unknownBlock.type.endsWith('_tool_result') && unknownBlock.type !== 'tool_result') {
        const serverResultBlock = block as unknown as ServerToolResultContent;
        const resultContent = extractServerToolResultContent(serverResultBlock.content);
        const toolName = toolNames.get(serverResultBlock.tool_use_id);
        
        return {
          type: 'element',
          tagName: 'server_tool_result',
          attributes: {
            name: toolName || unknownBlock.type,
            tool_type: unknownBlock.type.replace('_tool_result', ''),
          },
          children: [{ type: 'text', content: resultContent }],
        };
      }
      // Unknown block type
      return null;
    }
  }
}

/**
 * Convert Anthropic messages array to AgentNode[] for rendering.
 * 
 * Filters out the initial user message and converts assistant responses
 * and tool results to the AgentNode format used by the streaming renderer.
 * 
 * @param messages - Array of Anthropic API messages
 * @returns Array of AgentNodes for rendering
 */
export function convertMessagesToNodes(messages: unknown[]): AgentNode[] {
  const nodes: AgentNode[] = [];
  const toolNames: ToolNameMap = new Map();

  for (const msg of messages as Message[]) {
    // Skip user messages that are just the initial prompt
    // But include user messages with tool_result content (client or server)
    if (msg.role === 'user') {
      // Check if this user message contains tool results
      if (Array.isArray(msg.content)) {
        for (const block of msg.content as ContentBlock[]) {
          // Handle both client tool_result and server *_tool_result variants
          if (block.type === 'tool_result' || block.type.endsWith('_tool_result')) {
            const node = contentBlockToNode(block, toolNames);
            if (node) nodes.push(node);
          }
        }
      }
      continue;
    }

    // Process assistant messages
    if (msg.role === 'assistant') {
      const content = msg.content;
      
      if (typeof content === 'string') {
        // Simple string content
        nodes.push({ type: 'text', content });
      } else if (Array.isArray(content)) {
        // Array of content blocks
        for (const block of content as ContentBlock[]) {
          const node = contentBlockToNode(block, toolNames);
          if (node) nodes.push(node);
        }
      }
    }
  }

  return nodes;
}

/**
 * Check if a messages array has rich content (thinking, tools) worth converting.
 * If only simple text, we can skip conversion and use final_response directly.
 */
export function hasRichContent(messages: unknown[]): boolean {
  for (const msg of messages as Message[]) {
    if (msg.role === 'assistant' && Array.isArray(msg.content)) {
      for (const block of msg.content as ContentBlock[]) {
        // Check for thinking, client tools, and server tools
        if (
          block.type === 'thinking' || 
          block.type === 'tool_use' ||
          block.type === 'server_tool_use'
        ) {
          return true;
        }
      }
    }
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      for (const block of msg.content as ContentBlock[]) {
        // Check for both client tool_result and server *_tool_result variants
        if (block.type === 'tool_result' || block.type.endsWith('_tool_result')) {
          return true;
        }
      }
    }
  }
  return false;
}

