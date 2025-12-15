import { describe, it, expect, beforeEach } from 'vitest';
import {
  AnthropicStreamParser,
  type AnthropicEvent,
  type MessageStartEvent,
  type ContentBlockStartEvent,
  type ContentBlockDeltaEvent,
  type ContentBlockStopEvent,
} from '../../src/lib/parsers';
import { example_1 } from '../../src/lib/raw_stream_examples';

// Helper to create events
const createMessageStart = (): MessageStartEvent => ({
  type: 'message_start',
  message: {
    id: 'msg_test',
    type: 'message',
    role: 'assistant',
    model: 'claude-3',
    content: [],
    stop_reason: null,
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: 0 },
  },
});

const createBlockStart = (
  index: number,
  blockType: string,
  extra: Record<string, any> = {}
): ContentBlockStartEvent => ({
  type: 'content_block_start',
  index,
  content_block: { type: blockType, ...extra },
});

const createBlockDelta = (
  index: number,
  deltaType: 'text_delta' | 'thinking_delta' | 'input_json_delta',
  content: string
): ContentBlockDeltaEvent => {
  const delta: ContentBlockDeltaEvent['delta'] = { type: deltaType };
  if (deltaType === 'text_delta') delta.text = content;
  else if (deltaType === 'thinking_delta') delta.thinking = content;
  else if (deltaType === 'input_json_delta') delta.partial_json = content;
  return { type: 'content_block_delta', index, delta };
};

const createBlockStop = (index: number): ContentBlockStopEvent => ({
  type: 'content_block_stop',
  index,
});

describe('AnthropicStreamParser', () => {
  let parser: AnthropicStreamParser;

  beforeEach(() => {
    parser = new AnthropicStreamParser();
  });

  describe('integration test', () => {
    it('parses raw stream example correctly', () => {
      // Process all events in the example
      for (const item of example_1) {
        parser.processEvent(item as unknown as AnthropicEvent);
      }

      const nodes = parser.getNodes();

      // Verify we have the expected number of top-level blocks
      expect(nodes.length).toBe(9);

      // 1. Verify Thinking Block (Index 0)
      const thinkingNode = nodes[0];
      expect(thinkingNode.type).toBe('element');
      expect(thinkingNode.tagName).toBe('thinking');
      expect(thinkingNode.children?.[0].content).toContain(
        'The user is asking me to generate a sample invoice image'
      );

      // 2. Verify First Text Block (Index 1)
      const textNode1 = nodes[1];
      expect(textNode1.type).toBe('text');
      expect(textNode1.content).toContain("I'll create a new sample restaurant invoice image");

      // 3. Verify Server Tool Use (Index 2 - text_editor)
      const toolNode1 = nodes[2];
      expect(toolNode1.type).toBe('element');
      expect(toolNode1.tagName).toBe('server_tool_call');
      expect(toolNode1.attributes?.name).toBe('text_editor_code_execution');

      // Verify JSON content was parsed correctly
      const tool1Content = toolNode1.children?.[0].content || '';
      const tool1Input = JSON.parse(tool1Content);
      expect(tool1Input.command).toBe('create');
      expect(tool1Input.path).toBe('/tmp/create_restaurant_invoice.py');
      expect(tool1Input.file_text).toContain('from PIL import Image');

      // 4. Verify Tool Result (Index 3)
      const resultNode1 = nodes[3];
      expect(resultNode1.type).toBe('element');
      expect(resultNode1.tagName).toBe('text_editor_code_execution_tool_result');

      // 5. Verify Bash Tool Use (Index 4)
      const toolNode2 = nodes[4];
      expect(toolNode2.tagName).toBe('server_tool_call');
      expect(toolNode2.attributes?.name).toBe('bash_code_execution');
      const tool2Input = JSON.parse(toolNode2.children?.[0].content || '');
      expect(tool2Input.command).toBe('cd /tmp && python create_restaurant_invoice.py');

      // 6. Verify Bash Result (Index 5)
      const resultNode2 = nodes[5];
      expect(resultNode2.tagName).toBe('bash_code_execution_tool_result');
      const result2Content = JSON.parse(resultNode2.children?.[0].content || '');
      expect(result2Content.stdout).toContain('Restaurant invoice created successfully');

      // 7. Verify Final Text (Index 8)
      const finalTextNode = nodes[8];
      expect(finalTextNode.type).toBe('text');
      expect(finalTextNode.content).toContain(
        "Perfect! I've created a beautiful sample restaurant invoice"
      );
      expect(finalTextNode.content).toContain('Total: $205.29');
    });
  });

  describe('client tool_use', () => {
    it('handles client tool_use blocks correctly', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(
        createBlockStart(0, 'tool_use', { id: 'tool_123', name: 'calculator' })
      );
      parser.processEvent(createBlockDelta(0, 'input_json_delta', '{"a": 1,'));
      parser.processEvent(createBlockDelta(0, 'input_json_delta', ' "b": 2}'));
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);

      const toolNode = nodes[0];
      expect(toolNode.type).toBe('element');
      expect(toolNode.tagName).toBe('tool_call'); // client tool uses 'tool_call'
      expect(toolNode.attributes?.id).toBe('tool_123');
      expect(toolNode.attributes?.name).toBe('calculator');

      const parsedInput = JSON.parse(toolNode.children?.[0].content || '');
      expect(parsedInput).toEqual({ a: 1, b: 2 });
    });

    it('differentiates tool_call from server_tool_call', () => {
      parser.processEvent(createMessageStart());
      // Client tool
      parser.processEvent(createBlockStart(0, 'tool_use', { id: 'c1', name: 'client_tool' }));
      parser.processEvent(createBlockDelta(0, 'input_json_delta', '{}'));
      parser.processEvent(createBlockStop(0));
      // Server tool
      parser.processEvent(
        createBlockStart(1, 'server_tool_use', { id: 's1', name: 'server_tool' })
      );
      parser.processEvent(createBlockDelta(1, 'input_json_delta', '{}'));
      parser.processEvent(createBlockStop(1));

      const nodes = parser.getNodes();
      expect(nodes[0].tagName).toBe('tool_call');
      expect(nodes[1].tagName).toBe('server_tool_call');
    });
  });

  describe('error handling', () => {
    it('handles malformed JSON in tool input gracefully', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'tool_use', { id: 't1', name: 'test' }));
      parser.processEvent(createBlockDelta(0, 'input_json_delta', '{invalid json'));
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      // Should still create node, JSON remains unparsed string
      expect(nodes[0].children?.[0].content).toBe('{invalid json');
    });

    it('handles delta events for non-existent blocks', () => {
      parser.processEvent(createMessageStart());
      // Delta without corresponding block_start
      parser.processEvent(createBlockDelta(99, 'text_delta', 'orphan text'));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(0); // Should not crash, just ignore
    });

    it('handles stop events for non-existent blocks', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStop(99));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(0);
    });
  });

  describe('text with embedded XML', () => {
    it('parses text containing recognized tags like chart', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(
        createBlockDelta(0, 'text_delta', 'Here is a chart: <chart type="bar">data here</chart>')
      );
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      // parseMixedContent should extract the chart as an element (chart is whitelisted)
      expect(nodes.length).toBe(2);
      expect(nodes[0].type).toBe('text');
      expect(nodes[0].content).toBe('Here is a chart: ');
      expect(nodes[1].type).toBe('element');
      expect(nodes[1].tagName).toBe('chart');
      expect(nodes[1].attributes?.type).toBe('bar');
    });

    it('handles unclosed tags in streaming context', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(0, 'text_delta', 'Start <unclosed>content here'));
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      // Should still parse without crashing
      expect(nodes.length).toBeGreaterThan(0);
    });

    it('treats unrecognized XML tags as plain text', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(
        createBlockDelta(0, 'text_delta', '<outer><inner>nested</inner></outer>')
      );
      parser.processEvent(createBlockStop(0));

      // Unrecognized tags are preserved as plain text (for GFM/markdown rendering)
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].type).toBe('text');
      expect(nodes[0].content).toBe('<outer><inner>nested</inner></outer>');
    });
  });

  describe('edge cases', () => {
    it('clears state on message_start', () => {
      // First message
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(0, 'text_delta', 'First message'));
      parser.processEvent(createBlockStop(0));

      expect(parser.getNodes().length).toBe(1);

      // Second message_start should clear
      parser.processEvent(createMessageStart());

      expect(parser.getNodes().length).toBe(0);
    });

    it('returns empty array when no blocks exist', () => {
      const nodes = parser.getNodes();
      expect(nodes).toEqual([]);
    });

    it('handles empty text blocks', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(0); // parseMixedContent of '' returns []
    });

    it('handles blocks without content_block_stop', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(0, 'text_delta', 'Incomplete'));
      // No stop event

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].content).toBe('Incomplete');
    });

    it('handles multiple text blocks in order', () => {
      parser.processEvent(createMessageStart());

      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(0, 'text_delta', 'First'));
      parser.processEvent(createBlockStop(0));

      parser.processEvent(createBlockStart(1, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(1, 'text_delta', 'Second'));
      parser.processEvent(createBlockStop(1));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(2);
      expect(nodes[0].content).toBe('First');
      expect(nodes[1].content).toBe('Second');
    });

    it('preserves block order by index', () => {
      parser.processEvent(createMessageStart());

      // Add blocks out of order
      parser.processEvent(createBlockStart(2, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(2, 'text_delta', 'Third'));

      parser.processEvent(createBlockStart(0, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(0, 'text_delta', 'First'));

      parser.processEvent(createBlockStart(1, 'text', { text: '' }));
      parser.processEvent(createBlockDelta(1, 'text_delta', 'Second'));

      const nodes = parser.getNodes();
      expect(nodes[0].content).toBe('First');
      expect(nodes[1].content).toBe('Second');
      expect(nodes[2].content).toBe('Third');
    });
  });

  describe('fallback blocks', () => {
    it('handles unknown block types with rawBlock content as string', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(
        createBlockStart(0, 'custom_tool_result', {
          tool_use_id: 'tool_abc',
          content: 'Result string content',
        })
      );
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('custom_tool_result');
      expect(nodes[0].attributes?.tool_use_id).toBe('tool_abc');
      expect(nodes[0].children?.[0].content).toBe('Result string content');
    });

    it('handles tool results with object content', () => {
      const objContent = { status: 'success', data: [1, 2, 3] };
      parser.processEvent(createMessageStart());
      parser.processEvent(
        createBlockStart(0, 'web_search_tool_result', {
          tool_use_id: 'ws_123',
          content: objContent,
        })
      );
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('web_search_tool_result');

      const parsedContent = JSON.parse(nodes[0].children?.[0].content || '');
      expect(parsedContent).toEqual(objContent);
    });

    it('handles tool results with no content', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(
        createBlockStart(0, 'empty_tool_result', { tool_use_id: 'e1' })
      );
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].children?.[0].content).toBe('');
    });
  });

  describe('thinking blocks', () => {
    it('handles thinking blocks correctly', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(createBlockStart(0, 'thinking', { thinking: '' }));
      parser.processEvent(createBlockDelta(0, 'thinking_delta', 'Let me think...'));
      parser.processEvent(createBlockDelta(0, 'thinking_delta', ' about this.'));
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('thinking');
      expect(nodes[0].children?.[0].content).toBe('Let me think... about this.');
    });

    it('handles initial thinking content from block start', () => {
      parser.processEvent(createMessageStart());
      parser.processEvent(
        createBlockStart(0, 'thinking', { thinking: 'Initial thought' })
      );
      parser.processEvent(createBlockDelta(0, 'thinking_delta', ' and more'));
      parser.processEvent(createBlockStop(0));

      const nodes = parser.getNodes();
      expect(nodes[0].children?.[0].content).toBe('Initial thought and more');
    });
  });
});
