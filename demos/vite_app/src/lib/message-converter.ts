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

type ContentBlock = TextContent | ThinkingContent | ToolUseContent | ToolResultContent;

interface Message {
  role: 'user' | 'assistant';
  content: ContentBlock[] | string;
}

/**
 * Track tool names by their IDs so tool_result can reference them.
 */
type ToolNameMap = Map<string, string>;

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

    default:
      // Unknown block type - render as text if possible
      return null;
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
    // But include user messages with tool_result content
    if (msg.role === 'user') {
      // Check if this user message contains tool results
      if (Array.isArray(msg.content)) {
        for (const block of msg.content as ContentBlock[]) {
          if (block.type === 'tool_result') {
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
        if (block.type === 'thinking' || block.type === 'tool_use') {
          return true;
        }
      }
    }
    if (msg.role === 'user' && Array.isArray(msg.content)) {
      for (const block of msg.content as ContentBlock[]) {
        if (block.type === 'tool_result') {
          return true;
        }
      }
    }
  }
  return false;
}

