/**
 * String-level "mixed content" parser: turns a single text string into AgentNode[] by
 * recognizing only whitelisted XML-like tags (keeps other HTML-ish tags as plain text).
 * Used both by XmlStreamParser (XML stream format) and AnthropicStreamParser (embedded tags in raw text blocks).
 */
import type { AgentNode } from './types';
import { decodeHtmlEntities } from './utils';

/**
 * Whitelist of tag names that should be parsed as XML elements.
 * All other HTML-like tags are treated as plain text (passed to GFM renderer).
 * 
 * Includes:
 * - Agent structural tags: content-block-*, meta_init, etc.
 * - Custom rendering tags: chart, table (for special UI rendering)
 */
const RECOGNIZED_TAGS = new Set([
  // Agent structural tags
  'content-block-text',
  'content-block-thinking',
  'content-block-tool_call',
  'content-block-tool_result',
  'content-block-server_tool_call',
  'content-block-server_tool_result',
  'content-block-error',
  'content-block-meta_files',
  'meta_init',
  // Frontend tool relay tag
  'awaiting_frontend_tools',
  // Citation tags
  'citations',
  'citation',
  // Custom rendering tags within text blocks
  'chart',
  'table',
]);

/**
 * Check if a tag name should be parsed as an XML element.
 * Handles dynamic tool result tags (e.g., bash_code_execution_tool_result).
 */
function isRecognizedTag(tagName: string): boolean {
  if (RECOGNIZED_TAGS.has(tagName)) {
    return true;
  }
  // Handle dynamic *_tool_result tags
  if (tagName.endsWith('_tool_result')) {
    return true;
  }
  return false;
}

/**
 * Parses a string containing mixed text and XML-like tags into a tree structure.
 * Tolerant of unclosed tags (streaming context).
 * 
 * Only recognizes whitelisted tags as XML elements. Other HTML-like tags
 * (e.g., <details>, <kbd>, <div>) are preserved as plain text for GFM rendering.
 */
export function parseMixedContent(text: string): AgentNode[] {
  const root: AgentNode = { type: 'element', tagName: 'root', children: [] };
  const stack: AgentNode[] = [root];
  
  // Regex to match tags: <tag attr="val"> or </tag>
  // Captures: 1. full tag, 2. slash (if closing), 3. tag name, 4. attributes
  // Note: Tag names can include letters, digits, hyphens, underscores, and colons
  const tagRegex = /<(\/?)([a-zA-Z][a-zA-Z0-9_:-]*)([^>]*)>/g;
  
  let lastIndex = 0;
  let match;

  while ((match = tagRegex.exec(text)) !== null) {
    const [fullTag, slash, tagName, attributesStr] = match;
    const index = match.index;

    // Check if this is a recognized tag that should be parsed as XML
    if (!isRecognizedTag(tagName)) {
      // Unrecognized tag - skip it (will be included in text content)
      continue;
    }

    // Add preceding text as a text node
    if (index > lastIndex) {
      const textContent = text.slice(lastIndex, index);
      if (textContent) {
        const currentParent = stack[stack.length - 1];
        if (!currentParent.children) currentParent.children = [];
        currentParent.children.push({
          type: 'text',
          content: textContent
        });
      }
    }

    if (slash) {
      // Closing tag </tagName>
      // Find the matching opening tag in the stack (searching backwards)
      // If found, pop everything up to that point. If not found, ignore this closing tag (treat as text).
      let foundIndex = -1;
      for (let i = stack.length - 1; i > 0; i--) { // >0 because root is never popped
        if (stack[i].tagName === tagName) {
          foundIndex = i;
          break;
        }
      }

      if (foundIndex !== -1) {
        // Pop stack until we remove the matching tag
        while (stack.length > foundIndex) {
          stack.pop();
        }
      } else {
        // Treating orphan closing tag as text
        const currentParent = stack[stack.length - 1];
        if (!currentParent.children) currentParent.children = [];
        currentParent.children.push({
          type: 'text',
          content: fullTag
        });
      }
    } else {
      // Opening tag <tagName attributes>
      const attributes = parseAttributes(attributesStr);
      const newNode: AgentNode = {
        type: 'element',
        tagName: tagName,
        attributes,
        children: []
      };
      
      const currentParent = stack[stack.length - 1];
      if (!currentParent.children) currentParent.children = [];
      currentParent.children.push(newNode);
      
      // Push to stack to capture children
      // Self-closing tags logic could be added here if needed, but LLMs usually output <tag></tag>
      stack.push(newNode);
    }

    lastIndex = tagRegex.lastIndex;
  }

  // Add remaining text after the last tag
  if (lastIndex < text.length) {
    const textContent = text.slice(lastIndex);
    if (textContent) {
      const currentParent = stack[stack.length - 1];
      if (!currentParent.children) currentParent.children = [];
      currentParent.children.push({
        type: 'text',
        content: textContent
      });
    }
  }

  return root.children || [];
}

function parseAttributes(attrString: string): Record<string, string> {
  const attrs: Record<string, string> = {};
  // Note: underscore included for attributes like tool_use_id
  const attrRegex = /([a-zA-Z0-9_:-]+)=(?:"([^"]*)"|'([^']*)'|([^ \t\n\r\f/>]+))/g;
  let match;
  while ((match = attrRegex.exec(attrString)) !== null) {
    const key = match[1];
    const value = match[2] || match[3] || match[4] || "";
    attrs[key] = decodeHtmlEntities(value);
  }
  return attrs;
}

