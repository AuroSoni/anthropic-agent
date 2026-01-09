/**
 * Streaming parser for raw Anthropic JSON events: maintains per-content-block state and emits AgentNode[].
 * Note: for `text` blocks it also runs parseMixedContent() to support embedded XML-like tags inside streamed text.
 */
import type { AgentNode, AnthropicEvent, ContentBlockType, Citation } from './types';
import { parseMixedContent } from './xml-parser';

/**
 * Check if a block type represents a SERVER-side tool result.
 * Matches server_tool_use and dynamic types like bash_code_execution_tool_result.
 */
function isServerToolResultType(type: string): boolean {
  return type === 'server_tool_use' ||
         (type.endsWith('_tool_result') && type !== 'tool_result');
}

interface BlockState {
  type: ContentBlockType;
  content: string; // For text, thinking
  citations?: Citation[]; // Citations captured from content_block_stop
  toolData?: {
    id: string;
    name: string;
    inputJSON: string;
    input?: any;
  };
  // For generic blocks or tool results
  rawBlock?: any;
  isComplete: boolean;
}

/**
 * AnthropicStreamParser processes raw Anthropic streaming events.
 * 
 * It maintains internal state for each content block and converts
 * them to a normalized AgentNode[] tree structure.
 * 
 * Designed for aggressive real-time updates: getNodes() can be called
 * after every event to get the current tree state.
 */
export class AnthropicStreamParser {
  private blocks: Map<number, BlockState> = new Map();
  private blockOffset: number = 0;
  
  /**
   * Process a single raw event from the stream.
   */
  processEvent(event: AnthropicEvent): void {
    switch (event.type) {
      case 'message_start':
        // Calculate offset so new message blocks don't overwrite previous ones
        // This preserves blocks from earlier messages in multi-message streams
        // (e.g., when backend auto-executes client tools)
        if (this.blocks.size > 0) {
          this.blockOffset = Math.max(...this.blocks.keys()) + 1;
        }
        break;

      case 'content_block_start': {
        const { index, content_block } = event;
        const absoluteIndex = this.blockOffset + index;
        const blockState: BlockState = {
          type: content_block.type,
          content: '',
          isComplete: false,
          rawBlock: content_block
        };

        if (content_block.type === 'text') {
          blockState.content = content_block.text || '';
        } else if (content_block.type === 'thinking') {
          blockState.content = content_block.thinking || '';
        } else if (content_block.type === 'tool_use' || content_block.type === 'server_tool_use') {
          blockState.toolData = {
            id: content_block.id || '',
            name: content_block.name || '',
            inputJSON: ''
          };
        }

        this.blocks.set(absoluteIndex, blockState);
        break;
      }

      case 'content_block_delta': {
        const { index, delta } = event;
        const absoluteIndex = this.blockOffset + index;
        const block = this.blocks.get(absoluteIndex);
        if (!block) return;

        if (delta.type === 'text_delta' && delta.text) {
          block.content += delta.text;
        } else if (delta.type === 'thinking_delta' && delta.thinking) {
          block.content += delta.thinking;
        } else if (delta.type === 'input_json_delta' && delta.partial_json && block.toolData) {
          block.toolData.inputJSON += delta.partial_json;
        }
        break;
      }

      case 'content_block_stop': {
        const { index } = event;
        const absoluteIndex = this.blockOffset + index;
        const block = this.blocks.get(absoluteIndex);
        if (block) {
          block.isComplete = true;
          
          // Capture citations from the complete content block
          if (event.content_block?.citations) {
            block.citations = event.content_block.citations;
          }
          
          // Try to parse tool input JSON when block is complete
          if ((block.type === 'tool_use' || block.type === 'server_tool_use') && block.toolData) {
            try {
              block.toolData.input = JSON.parse(block.toolData.inputJSON);
            } catch (e) {
              // Partial or invalid JSON, leave as is
            }
          }
        }
        break;
      }
      
      case 'message_stop':
        // Final cleanup if needed
        break;
    }
  }

  /**
   * Convert current state to AgentNode[].
   * Can be called at any time for real-time updates.
   */
  getNodes(): AgentNode[] {
    const nodes: AgentNode[] = [];
    // Sort blocks by index to ensure order
    const sortedIndices = Array.from(this.blocks.keys()).sort((a, b) => a - b);

    for (const index of sortedIndices) {
      const block = this.blocks.get(index)!;

      if (block.type === 'text') {
        // Parse text content for mixed XML tags (artifacts, etc.)
        // Wrap in <text> element to match XmlStreamParser output structure
        const textChildren = parseMixedContent(block.content);
        nodes.push({
          type: 'element',
          tagName: 'text',
          children: textChildren
        });
        
        // Add citations node if present (matching XML parser structure)
        if (block.citations && block.citations.length > 0) {
          nodes.push(this.createCitationsNode(block.citations));
        }
      } else if (block.type === 'thinking') {
        nodes.push({
          type: 'element',
          tagName: 'thinking',
          children: [{ type: 'text', content: block.content }]
        });
      } else if (block.type === 'tool_use' || block.type === 'server_tool_use') {
        const toolData = block.toolData!;
        const attributes: Record<string, string> = {
          id: toolData.id,
          name: toolData.name,
        };
        
        let content = toolData.inputJSON;
        if (block.isComplete && toolData.input) {
            content = JSON.stringify(toolData.input, null, 2);
        }

        nodes.push({
          type: 'element',
          tagName: block.type === 'server_tool_use' ? 'server_tool_call' : 'tool_call',
          attributes,
          children: [{ type: 'text', content }]
        });
      } else {
        // Fallback for generic blocks (e.g. tool results)
        // If it has content content/json, stringify it
        let content = '';
        if (block.rawBlock?.content) {
           content = typeof block.rawBlock.content === 'string' 
             ? block.rawBlock.content 
             : JSON.stringify(block.rawBlock.content, null, 2);
        }
        
        const attributes: Record<string, string> = {};
        if (block.rawBlock?.tool_use_id) {
            attributes.id = block.rawBlock.tool_use_id;
        }

        // Normalize server tool results consistently with XmlStreamParser
        if (isServerToolResultType(block.type)) {
          const toolType = block.type.replace(/_tool_result$/, '');
          nodes.push({
            type: 'element',
            tagName: 'server_tool_result',
            attributes: {
              ...attributes,
              name: block.type,
              toolType: toolType,
            },
            children: [{ type: 'text', content }]
          });
        } else {
          nodes.push({
            type: 'element',
            tagName: block.type,
            attributes: {
              ...attributes,
              name: block.rawBlock?.name || block.type,
            },
            children: [{ type: 'text', content }]
          });
        }
      }
    }

    return nodes;
  }

  /**
   * Create a citations node matching the XML parser's structure.
   */
  private createCitationsNode(citations: Citation[]): AgentNode {
    return {
      type: 'element',
      tagName: 'citations',
      attributes: {},
      children: citations.map(c => ({
        type: 'element' as const,
        tagName: 'citation',
        attributes: this.buildCitationAttrs(c),
        children: [{ type: 'text' as const, content: c.cited_text || '' }]
      }))
    };
  }

  /**
   * Build citation attributes, only including non-empty values.
   */
  private buildCitationAttrs(c: Citation): Record<string, string> {
    const attrs: Record<string, string> = {};
    if (c.type) attrs.type = c.type;
    if (c.url) attrs.url = c.url;
    if (c.title) attrs.title = c.title;
    if (c.document_index) attrs.document_index = c.document_index;
    if (c.start_page_number) attrs.start_page_number = c.start_page_number;
    if (c.end_page_number) attrs.end_page_number = c.end_page_number;
    return attrs;
  }

  /**
   * Reset the parser state.
   */
  reset(): void {
    this.blocks.clear();
    this.blockOffset = 0;
  }
}

