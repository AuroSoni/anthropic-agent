import { describe, it, expect, beforeEach } from 'vitest';
import { JsonStreamParser } from '../../src/lib/parsers/json-stream-parser';
import type { JsonEnvelope } from '../../src/lib/parsers/json-types';
import { findNodeByTag, getNodeTextContent, hasExpectedNodeTypes } from '../fixtures/helpers';
import {
  SIMPLE_TEXT,
  TOOL_CALL,
  CHUNKED_TOOL_CALL,
  SERVER_TOOL,
  CITATIONS,
  TOOL_RESULT_IMAGE,
  AWAITING_FRONTEND_TOOLS,
  ERROR_ENVELOPE,
  PDF_CITATIONS,
  CHAR_LOCATION_CITATIONS,
  META_FILES,
  MULTI_BLOCK,
} from '../fixtures/json-scenarios';

describe('JsonStreamParser', () => {
  let parser: JsonStreamParser;

  beforeEach(() => {
    parser = new JsonStreamParser();
  });

  /** Feed all envelopes (skipping meta_init/meta_final which are handled upstream). */
  function feedAll(envelopes: JsonEnvelope[]) {
    for (const env of envelopes) {
      if (env.type === 'meta_init' || env.type === 'meta_final') continue;
      parser.processEnvelope(env);
    }
  }

  // =========================================================================
  // Basic streaming
  // =========================================================================

  describe('thinking and text blocks', () => {
    it('parses thinking + text from SIMPLE_TEXT fixture', () => {
      feedAll(SIMPLE_TEXT);
      const nodes = parser.getNodes();

      expect(hasExpectedNodeTypes(nodes, ['thinking', 'text'])).toBe(true);

      const thinking = findNodeByTag(nodes, 'thinking')!;
      expect(getNodeTextContent(thinking)).toBe('Let me think about this.');

      const text = findNodeByTag(nodes, 'text')!;
      expect(getNodeTextContent(text)).toBe('Hello! How can I help?');
    });

    it('handles multiple thinking + text pairs', () => {
      feedAll(MULTI_BLOCK);
      const nodes = parser.getNodes();

      expect(nodes.length).toBe(4);
      expect(nodes[0].tagName).toBe('thinking');
      expect(getNodeTextContent(nodes[0])).toBe('First thought');
      expect(nodes[1].tagName).toBe('text');
      expect(getNodeTextContent(nodes[1])).toBe('First response');
      expect(nodes[2].tagName).toBe('thinking');
      expect(getNodeTextContent(nodes[2])).toBe('Second thought');
      expect(nodes[3].tagName).toBe('text');
      expect(getNodeTextContent(nodes[3])).toBe('Second response');
    });

    it('supports real-time updates during streaming', () => {
      // Partial thinking
      parser.processEnvelope({ type: 'thinking', agent: 'a', final: false, delta: 'Hmm' });
      let nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(getNodeTextContent(nodes[0])).toBe('Hmm');

      // More thinking
      parser.processEnvelope({ type: 'thinking', agent: 'a', final: false, delta: '...' });
      nodes = parser.getNodes();
      expect(getNodeTextContent(nodes[0])).toBe('Hmm...');

      // Close thinking
      parser.processEnvelope({ type: 'thinking', agent: 'a', final: true, delta: '' });
      nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(getNodeTextContent(nodes[0])).toBe('Hmm...');
    });
  });

  // =========================================================================
  // Tool calls
  // =========================================================================

  describe('tool calls', () => {
    it('parses tool_call + tool_result from TOOL_CALL fixture', () => {
      feedAll(TOOL_CALL);
      const nodes = parser.getNodes();

      expect(hasExpectedNodeTypes(nodes, ['thinking', 'tool_call', 'tool_result', 'text'])).toBe(true);

      const toolCall = findNodeByTag(nodes, 'tool_call')!;
      expect(toolCall.attributes?.id).toBe('tool_001');
      expect(toolCall.attributes?.name).toBe('calculator');
      const inputJson = JSON.parse(getNodeTextContent(toolCall));
      expect(inputJson.expression).toBe('5 * (3 + 2/4)');

      const toolResult = findNodeByTag(nodes, 'tool_result')!;
      expect(toolResult.attributes?.id).toBe('tool_001');
      expect(getNodeTextContent(toolResult)).toBe('17.5');
    });

    it('handles chunked tool_call deltas', () => {
      feedAll(CHUNKED_TOOL_CALL);
      const nodes = parser.getNodes();

      const toolCall = findNodeByTag(nodes, 'tool_call')!;
      expect(toolCall.attributes?.id).toBe('tool_002');
      const inputJson = JSON.parse(getNodeTextContent(toolCall));
      expect(inputJson.expression).toBe('1+1');
    });
  });

  // =========================================================================
  // Server tools
  // =========================================================================

  describe('server tools', () => {
    it('parses server_tool_call + server_tool_result from SERVER_TOOL fixture', () => {
      feedAll(SERVER_TOOL);
      const nodes = parser.getNodes();

      expect(hasExpectedNodeTypes(nodes, ['thinking', 'server_tool_call', 'server_tool_result', 'text'])).toBe(true);

      const stc = findNodeByTag(nodes, 'server_tool_call')!;
      expect(stc.attributes?.name).toBe('web_search');
      const stcInput = JSON.parse(getNodeTextContent(stc));
      expect(stcInput.query).toBe('latest YC news');

      const str = findNodeByTag(nodes, 'server_tool_result')!;
      expect(str.attributes?.name).toBe('web_search');
      const strContent = JSON.parse(getNodeTextContent(str));
      expect(strContent[0].title).toBe('YC News');
    });
  });

  // =========================================================================
  // Citations
  // =========================================================================

  describe('citations', () => {
    it('attaches citations to preceding text block', () => {
      feedAll(CITATIONS);
      const nodes = parser.getNodes();

      expect(hasExpectedNodeTypes(nodes, ['text', 'citations'])).toBe(true);

      const text = findNodeByTag(nodes, 'text')!;
      expect(getNodeTextContent(text)).toContain('Rayleigh scattering');

      const citations = findNodeByTag(nodes, 'citations')!;
      expect(citations.children?.length).toBe(2);

      const c1 = citations.children![0];
      expect(c1.tagName).toBe('citation');
      expect(c1.attributes?.url).toBe('https://example.com/sky');
      expect(getNodeTextContent(c1)).toBe('Rayleigh scattering');

      const c2 = citations.children![1];
      expect(getNodeTextContent(c2)).toBe('blue light wavelength');
    });

    it('parses PDF citations with page numbers', () => {
      feedAll(PDF_CITATIONS);
      const nodes = parser.getNodes();

      const citations = findNodeByTag(nodes, 'citations')!;
      expect(citations.children?.length).toBe(2);

      const c1 = citations.children![0];
      expect(c1.attributes?.type).toBe('page_location');
      expect(c1.attributes?.start_page_number).toBe('3');
      expect(c1.attributes?.end_page_number).toBe('4');
    });

    it('parses char-location citations with char indices', () => {
      feedAll(CHAR_LOCATION_CITATIONS);
      const nodes = parser.getNodes();

      const citations = findNodeByTag(nodes, 'citations')!;
      expect(citations.children?.length).toBe(2);

      const c1 = citations.children![0];
      expect(c1.attributes?.type).toBe('char_location');
      expect(c1.attributes?.document_title).toBe('My Document');
      expect(c1.attributes?.start_char_index).toBe('0');
      expect(c1.attributes?.end_char_index).toBe('20');
      expect(getNodeTextContent(c1)).toBe('The grass is green. ');

      const c2 = citations.children![1];
      expect(c2.attributes?.start_char_index).toBe('20');
      expect(c2.attributes?.end_char_index).toBe('36');
    });
  });

  // =========================================================================
  // Tool result with image
  // =========================================================================

  describe('tool result with image', () => {
    it('emits image nodes after tool_result', () => {
      feedAll(TOOL_RESULT_IMAGE);
      const nodes = parser.getNodes();

      expect(hasExpectedNodeTypes(nodes, ['tool_call', 'tool_result', 'image'])).toBe(true);

      const image = findNodeByTag(nodes, 'image')!;
      expect(image.attributes?.src).toBe('data:image/png;base64,iVBORw0KGgo=');
      expect(image.attributes?.media_type).toBe('image/png');
    });
  });

  // =========================================================================
  // Awaiting frontend tools
  // =========================================================================

  describe('awaiting frontend tools', () => {
    it('creates awaiting_frontend_tools node with data attribute', () => {
      feedAll(AWAITING_FRONTEND_TOOLS);
      const nodes = parser.getNodes();

      expect(hasExpectedNodeTypes(nodes, ['thinking', 'tool_call', 'awaiting_frontend_tools'])).toBe(true);

      const awaiting = findNodeByTag(nodes, 'awaiting_frontend_tools')!;
      const data = JSON.parse(awaiting.attributes!.data);
      expect(data).toHaveLength(1);
      expect(data[0].name).toBe('user_confirm');
      expect(data[0].tool_use_id).toBe('tool_004');
    });
  });

  // =========================================================================
  // Error handling
  // =========================================================================

  describe('error handling', () => {
    it('creates error node from error envelope', () => {
      feedAll(ERROR_ENVELOPE);
      const nodes = parser.getNodes();

      expect(nodes.length).toBe(1);
      const error = findNodeByTag(nodes, 'error')!;
      const errorData = JSON.parse(getNodeTextContent(error));
      expect(errorData.message).toBe('Rate limit exceeded');
      expect(errorData.code).toBe(429);
    });
  });

  // =========================================================================
  // Meta files
  // =========================================================================

  describe('meta files', () => {
    it('creates meta_files node with file data', () => {
      feedAll(META_FILES);
      const nodes = parser.getNodes();

      expect(hasExpectedNodeTypes(nodes, ['thinking', 'text', 'meta_files'])).toBe(true);

      const metaFiles = findNodeByTag(nodes, 'meta_files')!;
      const filesData = JSON.parse(getNodeTextContent(metaFiles));
      expect(filesData.files).toHaveLength(1);
      expect(filesData.files[0].filename).toBe('monthly_budget.xlsx');
      expect(filesData.files[0].file_id).toBe('file_011CYE1GuHJ546z4Z8dUwqaF');
    });
  });

  // =========================================================================
  // Meta init / meta final passthrough
  // =========================================================================

  describe('meta passthrough', () => {
    it('ignores meta_init and meta_final envelopes', () => {
      parser.processEnvelope({ type: 'meta_init', agent: 'a', final: true, delta: '{}' });
      parser.processEnvelope({ type: 'meta_final', agent: 'a', final: true, delta: '{}' });
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(0);
    });
  });

  // =========================================================================
  // Reset
  // =========================================================================

  describe('reset', () => {
    it('clears all state', () => {
      feedAll(SIMPLE_TEXT);
      expect(parser.getNodes().length).toBeGreaterThan(0);

      parser.reset();
      expect(parser.getNodes().length).toBe(0);
    });
  });

  // =========================================================================
  // Edge cases
  // =========================================================================

  describe('edge cases', () => {
    it('returns empty array when no envelopes processed', () => {
      expect(parser.getNodes()).toEqual([]);
    });

    it('handles unknown envelope types gracefully', () => {
      parser.processEnvelope({ type: 'unknown_future_type', agent: 'a', final: true, delta: 'data' });
      expect(parser.getNodes()).toEqual([]);
    });

    it('handles empty text delta without creating empty node', () => {
      parser.processEnvelope({ type: 'text', agent: 'a', final: true, delta: '' });
      // final:true with empty delta and no open block → no node created
      expect(parser.getNodes().length).toBe(0);
    });

    it('handles text with embedded XML tags', () => {
      parser.processEnvelope({ type: 'text', agent: 'a', final: false, delta: 'Chart: <chart type="bar">data</chart>' });
      parser.processEnvelope({ type: 'text', agent: 'a', final: true, delta: '' });

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      const text = nodes[0];
      expect(text.tagName).toBe('text');
      // parseMixedContent should split into text + chart element
      expect(text.children!.length).toBe(2);
      expect(text.children![0].type).toBe('text');
      expect(text.children![0].content).toBe('Chart: ');
      expect(text.children![1].type).toBe('element');
      expect(text.children![1].tagName).toBe('chart');
    });

    it('handles citations without preceding text block', () => {
      parser.processEnvelope({
        type: 'citation', agent: 'a', final: true, delta: 'cited text',
        citation_type: 'webpage', url: 'https://example.com',
      });
      const nodes = parser.getNodes();
      // Should create a standalone text block with empty content + citations
      expect(nodes.length).toBe(2); // empty text + citations
      expect(nodes[1].tagName).toBe('citations');
    });

    it('handles malformed JSON in tool_call delta', () => {
      parser.processEnvelope({ type: 'tool_call', agent: 'a', final: true, delta: '{bad json', id: 't1', name: 'test' });
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      // Should keep raw content when JSON parse fails
      expect(getNodeTextContent(nodes[0])).toBe('{bad json');
    });
  });
});
