/**
 * Streaming XML-format parser (backend `<content-block-*>` style): buffers chunks, handles CDATA safely,
 * and normalizes tags/attributes into the same AgentNode[] shape used by the raw (Anthropic event) parser.
 * This is the main entrypoint when consuming the "xml" stream format.
 */
import type { AgentNode } from './types';
import { parseMixedContent } from './xml-parser';
import { decodeHtmlEntities, unescapeSseNewlines } from './utils';

/**
 * Tag name mapping from XML format (content-block-*) to normalized names.
 * Maps backend XML tags to standard node tag names matching AnthropicStreamParser output.
 */
const TAG_NAME_MAP: Record<string, string> = {
  'content-block-text': 'text',
  'content-block-thinking': 'thinking',
  'content-block-tool_call': 'tool_call',
  'content-block-tool_result': 'tool_result',
  'content-block-server_tool_call': 'server_tool_call',
  'content-block-server_tool_result': 'server_tool_result',
  'content-block-error': 'error',
  'content-block-meta_files': 'meta_files',
  'meta_init': 'meta_init',
};

/**
 * Check if a tag name represents a SERVER-side tool result.
 * Matches server_tool_result and dynamic types like bash_code_execution_tool_result.
 * Excludes client-side tool_result and content-block-tool_result.
 */
function isServerToolResultTag(tagName: string): boolean {
  return (tagName === 'server_tool_result' || 
          tagName === 'content-block-server_tool_result' ||
          (tagName.endsWith('_tool_result') && 
           tagName !== 'tool_result' && 
           tagName !== 'content-block-tool_result'));
}

/**
 * Check if a tag name represents a CLIENT-side tool result.
 */
function isClientToolResultTag(tagName: string): boolean {
  return tagName === 'tool_result' || tagName === 'content-block-tool_result';
}

/**
 * Escape special regex characters in a string.
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Pre-process CDATA sections by extracting their content.
 * CDATA content is preserved as-is (not escaped).
 * Detects incomplete CDATA at buffer end for streaming scenarios.
 */
function preprocessCDATA(text: string): { 
  processed: string; 
  cdataMap: Map<string, string>;
  incompleteStart: number;
} {
  const cdataMap = new Map<string, string>();
  let counter = 0;
  
  // Check for incomplete CDATA: <![CDATA[... without ]]>
  const incompleteMatch = text.match(/<!\[CDATA\[(?:(?!\]\]>)[\s\S])*$/);
  const incompleteStart = incompleteMatch ? incompleteMatch.index! : -1;
  
  // Only process complete portion (exclude incomplete CDATA at end)
  const textToProcess = incompleteStart >= 0 ? text.slice(0, incompleteStart) : text;
  
  // Match complete CDATA sections: <![CDATA[...]]>
  const processed = textToProcess.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, (_, content) => {
    const placeholder = `__CDATA_PLACEHOLDER_${counter++}__`;
    cdataMap.set(placeholder, content);
    return placeholder;
  });
  
  return { processed, cdataMap, incompleteStart };
}

/**
 * Restore CDATA placeholders with their original content.
 */
function restoreCDATA(text: string, cdataMap: Map<string, string>): string {
  let result = text;
  for (const [placeholder, content] of cdataMap) {
    result = result.replace(new RegExp(escapeRegex(placeholder), 'g'), content);
  }
  return result;
}

/**
 * Normalize tag names in an AgentNode tree.
 * Converts content-block-* tags to standard names and handles nested structures.
 * Also unescapes SSE newlines in text content for display.
 */
function normalizeNodes(nodes: AgentNode[], cdataMap: Map<string, string>): AgentNode[] {
  return nodes.map(node => {
    if (node.type === 'text') {
      // Restore CDATA content AND unescape SSE newlines for display
      const restored = node.content ? restoreCDATA(node.content, cdataMap) : '';
      return {
        ...node,
        content: unescapeSseNewlines(restored)
      };
    }
    
    if (node.type === 'element' && node.tagName) {
      // Normalize tag name via mapping
      const normalizedTag = TAG_NAME_MAP[node.tagName] || node.tagName;
      
      // Handle tool_call/server_tool_call: parse arguments attribute if present
      if ((normalizedTag === 'tool_call' || normalizedTag === 'server_tool_call') && node.attributes?.arguments) {
        try {
          // Decode HTML entities before parsing JSON (JSON handles \n natively)
          const decodedArgs = decodeHtmlEntities(node.attributes.arguments);
          const parsedArgs = JSON.parse(decodedArgs);
          // Replace arguments string with formatted JSON as child content
          return {
            type: 'element',
            tagName: normalizedTag, // Preserve: 'tool_call' or 'server_tool_call'
            attributes: {
              id: node.attributes.id || '',
              name: node.attributes.name || '',
            },
            children: [{ type: 'text', content: JSON.stringify(parsedArgs, null, 2) }]
          };
        } catch {
          // Keep as-is if parsing fails
        }
      }
      
      // Handle server tool results: normalize to server_tool_result with toolType
      if (isServerToolResultTag(node.tagName)) {
        // Extract tool type from dynamic names like "bash_code_execution_tool_result"
        let toolType = node.attributes?.name || '';
        if (!toolType && node.tagName.endsWith('_tool_result')) {
          toolType = node.tagName
            .replace(/_tool_result$/, '')
            .replace(/^content-block-/, '');
        }
        return {
          type: 'element',
          tagName: 'server_tool_result',
          attributes: {
            id: node.attributes?.id || '',
            name: node.attributes?.name || 'server_tool_result',
            toolType: toolType,
          },
          children: node.children ? normalizeNodes(node.children, cdataMap) : undefined
        };
      }
      
      // Handle client tool_result
      if (isClientToolResultTag(node.tagName)) {
        return {
          type: 'element',
          tagName: 'tool_result',
          attributes: {
            id: node.attributes?.id || '',
            name: node.attributes?.name || 'tool_result',
          },
          children: node.children ? normalizeNodes(node.children, cdataMap) : undefined
        };
      }
      
      return {
        ...node,
        tagName: normalizedTag,
        children: node.children ? normalizeNodes(node.children, cdataMap) : undefined
      };
    }
    
    return node;
  });
}

/**
 * XmlStreamParser processes XML-formatted streaming output from the agent.
 * 
 * It accumulates text chunks, handles CDATA sections, and normalizes
 * content-block-* tags to standard tag names matching AnthropicStreamParser output.
 * 
 * Designed for aggressive real-time updates: getNodes() can be called after
 * every chunk to get the current tree state (even with partial/unclosed tags).
 */
export class XmlStreamParser {
  private buffer: string = '';
  
  /**
   * Append a chunk of XML text to the buffer.
   */
  appendChunk(chunk: string): void {
    this.buffer += chunk;
  }
  
  /**
 * Get the current node tree.
 * Can be called at any time for real-time updates.
 * Tolerant of unclosed tags (uses parseMixedContent which handles streaming).
 * Defers parsing of incomplete CDATA sections at buffer end.
   */
  getNodes(): AgentNode[] {
    if (!this.buffer) {
      return [];
    }
    
    // Pre-process CDATA sections (excludes incomplete CDATA at end)
    const { processed, cdataMap } = preprocessCDATA(this.buffer);
    
    // Parse only the complete portion of XML structure
    const rawNodes = parseMixedContent(processed);
    
    // Normalize tag names and restore CDATA content
    return normalizeNodes(rawNodes, cdataMap);
  }
  
  /**
   * Reset the parser state.
   */
  reset(): void {
    this.buffer = '';
  }
  
  /**
   * Get the raw buffer content (for debugging).
   */
  getRawContent(): string {
    return this.buffer;
  }
}

