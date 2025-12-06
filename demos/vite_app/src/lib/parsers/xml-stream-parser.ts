import type { AgentNode } from './types';
import { parseMixedContent } from './xml-parser';

/**
 * Tag name mapping from XML format (content-block-*) to normalized names.
 * Maps backend XML tags to standard node tag names matching AnthropicStreamParser output.
 */
const TAG_NAME_MAP: Record<string, string> = {
  'content-block-text': 'text',
  'content-block-thinking': 'thinking',
  'content-block-tool_call': 'tool_call',
  'content-block-tool_result': 'tool_result',
  'content-block-error': 'error',
  'content-block-meta_files': 'meta_files',
  'meta_init': 'meta_init',
};

/**
 * Escape special regex characters in a string.
 */
function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Pre-process CDATA sections by extracting their content.
 * CDATA content is preserved as-is (not escaped).
 */
function preprocessCDATA(text: string): { processed: string; cdataMap: Map<string, string> } {
  const cdataMap = new Map<string, string>();
  let counter = 0;
  
  // Match CDATA sections: <![CDATA[...]]>
  const processed = text.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, (_, content) => {
    const placeholder = `__CDATA_PLACEHOLDER_${counter++}__`;
    cdataMap.set(placeholder, content);
    return placeholder;
  });
  
  return { processed, cdataMap };
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
 */
function normalizeNodes(nodes: AgentNode[], cdataMap: Map<string, string>): AgentNode[] {
  return nodes.map(node => {
    if (node.type === 'text') {
      // Restore CDATA content in text nodes
      return {
        ...node,
        content: node.content ? restoreCDATA(node.content, cdataMap) : node.content
      };
    }
    
    if (node.type === 'element' && node.tagName) {
      // Normalize tag name
      const normalizedTag = TAG_NAME_MAP[node.tagName] || node.tagName;
      
      // Handle tool_call: parse arguments attribute if present
      if (normalizedTag === 'tool_call' && node.attributes?.arguments) {
        try {
          const parsedArgs = JSON.parse(node.attributes.arguments);
          // Replace arguments string with formatted JSON as child content
          return {
            type: 'element',
            tagName: normalizedTag,
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
   */
  getNodes(): AgentNode[] {
    if (!this.buffer) {
      return [];
    }
    
    // Pre-process CDATA sections
    const { processed, cdataMap } = preprocessCDATA(this.buffer);
    
    // Parse the XML structure
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

