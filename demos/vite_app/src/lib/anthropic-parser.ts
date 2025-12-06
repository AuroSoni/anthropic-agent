import { type AgentNode, parseMixedContent } from './xml-parser';
import type {
  AnthropicEvent,
  ContentBlockType
} from './anthropic-types';

interface BlockState {
  type: ContentBlockType;
  content: string; // For text, thinking
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

export class AnthropicStreamParser {
  private blocks: Map<number, BlockState> = new Map();
  
  // Process a single raw event from the stream
  processEvent(event: AnthropicEvent): void {
    switch (event.type) {
      case 'message_start':
        // Reset state if needed, or just ignore if we're only handling content
        this.blocks.clear();
        break;

      case 'content_block_start': {
        const { index, content_block } = event;
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

        this.blocks.set(index, blockState);
        break;
      }

      case 'content_block_delta': {
        const { index, delta } = event;
        const block = this.blocks.get(index);
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
        const block = this.blocks.get(index);
        if (block) {
          block.isComplete = true;
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

  // Convert current state to AgentNode[]
  getNodes(): AgentNode[] {
    const nodes: AgentNode[] = [];
    // Sort blocks by index to ensure order
    const sortedIndices = Array.from(this.blocks.keys()).sort((a, b) => a - b);

    for (const index of sortedIndices) {
      const block = this.blocks.get(index)!;

      if (block.type === 'text') {
        // Parse text content for mixed XML tags (artifacts, etc.)
        nodes.push(...parseMixedContent(block.content));
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
            attributes.tool_use_id = block.rawBlock.tool_use_id;
        }

        nodes.push({
            type: 'element',
            tagName: block.type, // Use the raw type as tag name (e.g. text_editor_code_execution_tool_result)
            attributes,
            children: [{ type: 'text', content }]
        });
      }
    }

    return nodes;
  }
}
