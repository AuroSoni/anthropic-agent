import type { AgentNode } from './types';

/**
 * Parses a string containing mixed text and XML-like tags into a tree structure.
 * Tolerant of unclosed tags (streaming context).
 */
export function parseMixedContent(text: string): AgentNode[] {
  const root: AgentNode = { type: 'element', tagName: 'root', children: [] };
  const stack: AgentNode[] = [root];
  
  // Regex to match tags: <tag attr="val"> or </tag>
  // Captures: 1. full tag, 2. slash (if closing), 3. tag name, 4. attributes
  const tagRegex = /<(\/?)([a-zA-Z][a-zA-Z0-9:-]*)([^>]*)>/g;
  
  let lastIndex = 0;
  let match;

  while ((match = tagRegex.exec(text)) !== null) {
    const [fullTag, slash, tagName, attributesStr] = match;
    const index = match.index;

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
  const attrRegex = /([a-zA-Z0-9:-]+)=(?:"([^"]*)"|'([^']*)'|([^ \t\n\r\f/>]+))/g;
  let match;
  while ((match = attrRegex.exec(attrString)) !== null) {
    const key = match[1];
    const value = match[2] || match[3] || match[4] || "";
    attrs[key] = value;
  }
  return attrs;
}

