import { describe, it, expect, beforeEach } from 'vitest';
import { XmlStreamParser } from '../../src/lib/parsers';
import { xml_example_1_raw, cleanXmlStream } from '../../src/lib/xml_stream_examples';
import { HELLO_XML, CLIENT_TOOLS_XML, SERVER_TOOLS_XML, CITATIONS_XML } from '../fixtures/scenarios';
import { extractSseData, hasExpectedNodeTypes } from '../fixtures/helpers';

describe('XmlStreamParser', () => {
  let parser: XmlStreamParser;

  beforeEach(() => {
    parser = new XmlStreamParser();
  });

  describe('integration test', () => {
    it('parses full XML stream example correctly', () => {
      // Clean the SSE format and feed to parser
      const cleanedXml = cleanXmlStream(xml_example_1_raw);
      parser.appendChunk(cleanedXml);

      const nodes = parser.getNodes();

      // Expected structure from xml_stream_examples.txt:
      // 1. meta_init
      // 2. thinking
      // 3. text (first response)
      // 4-5. tool_call + tool_result (text_editor_code_execution - create first script)
      // 6-7. tool_call + tool_result (bash_code_execution - run first script, fails)
      // 8. text (explanation)
      // 9-10. tool_call + tool_result (bash_code_execution - check modules)
      // 11-12. tool_call + tool_result (bash_code_execution - check chart method)
      // 13. text (explanation)
      // 14-15. tool_call + tool_result (text_editor_code_execution - create v2 script)
      // 16-17. tool_call + tool_result (bash_code_execution - run v2 script, succeeds)
      // 18. text (final response)
      // 19. meta_files

      // Verify we have all expected nodes
      expect(nodes.length).toBe(19);

      // 1. Verify meta_init
      const metaInitNode = nodes[0];
      expect(metaInitNode.type).toBe('element');
      expect(metaInitNode.tagName).toBe('meta_init');
      expect(metaInitNode.attributes?.data).toContain('format');
      expect(metaInitNode.attributes?.data).toContain('Create a docx file with a chart');

      // 2. Verify thinking block
      const thinkingNode = nodes[1];
      expect(thinkingNode.type).toBe('element');
      expect(thinkingNode.tagName).toBe('thinking');
      expect(thinkingNode.children?.[0].content).toContain('The user wants me to create a');
      expect(thinkingNode.children?.[0].content).toContain('docx file with a chart');

      // 3. Verify first text block (normalized from content-block-text)
      const textNode1 = nodes[2];
      expect(textNode1.type).toBe('element');
      expect(textNode1.tagName).toBe('text');
      const text1Content = textNode1.children?.[0].content || '';
      expect(text1Content).toContain("I'll create");
      expect(text1Content).toContain('python-docx library');

      // 4. Verify first tool_call (text_editor_code_execution)
      const toolCall1 = nodes[3];
      expect(toolCall1.type).toBe('element');
      expect(toolCall1.tagName).toBe('tool_call');
      expect(toolCall1.attributes?.id).toBe('srvtoolu_01AhsGyKMBXkHzNR43LqDeMy');
      expect(toolCall1.attributes?.name).toBe('text_editor_code_execution');
      // Verify arguments are parsed as JSON in children
      const tool1Args = JSON.parse(toolCall1.children?.[0].content || '{}');
      expect(tool1Args.command).toBe('create');
      expect(tool1Args.path).toBe('/tmp/create_docx_with_chart.py');

      // 5. Verify first tool_result (normalized from text_editor_code_execution_tool_result)
      const toolResult1 = nodes[4];
      expect(toolResult1.type).toBe('element');
      expect(toolResult1.tagName).toBe('tool_result');
      expect(toolResult1.attributes?.id).toBe('srvtoolu_01AhsGyKMBXkHzNR43LqDeMy');
      expect(toolResult1.attributes?.name).toBe('text_editor_code_execution_tool_result');
      // CDATA content should be preserved
      expect(toolResult1.children?.[0].content).toContain('BetaTextEditorCodeExecutionCreateResultBlock');

      // 6. Verify second tool_call (bash_code_execution)
      const toolCall2 = nodes[5];
      expect(toolCall2.tagName).toBe('tool_call');
      expect(toolCall2.attributes?.name).toBe('bash_code_execution');
      const tool2Args = JSON.parse(toolCall2.children?.[0].content || '{}');
      expect(tool2Args.command).toContain('python create_docx_with_chart.py');

      // 7. Verify bash tool_result with error (CDATA content)
      const toolResult2 = nodes[6];
      expect(toolResult2.tagName).toBe('tool_result');
      expect(toolResult2.children?.[0].content).toContain('return_code=1');
      expect(toolResult2.children?.[0].content).toContain('ModuleNotFoundError');

      // 8. Verify explanatory text
      const textNode2 = nodes[7];
      expect(textNode2.type).toBe('element');
      expect(textNode2.tagName).toBe('text');
      const text2Content = textNode2.children?.[0].content || '';
      expect(text2Content).toContain('Let me check');
      expect(text2Content).toContain('simpler version');

      // 13. Verify text explaining chart support issue
      const textNode3 = nodes[12];
      expect(textNode3.type).toBe('element');
      expect(textNode3.tagName).toBe('text');
      const text3Content = textNode3.children?.[0].content || '';
      expect(text3Content).toContain("python-docx library doesn't have built-in chart support");

      // 18. Verify final text block
      const finalTextNode = nodes[17];
      expect(finalTextNode.type).toBe('element');
      expect(finalTextNode.tagName).toBe('text');
      const finalTextContent = finalTextNode.children?.[0].content || '';
      expect(finalTextContent).toContain('Perfect');
      expect(finalTextContent).toContain("document_with_chart.docx");
      expect(finalTextContent).toContain('ready for download');

      // 19. Verify meta_files block (CDATA content)
      const metaFilesNode = nodes[18];
      expect(metaFilesNode.type).toBe('element');
      expect(metaFilesNode.tagName).toBe('meta_files');
      const metaFilesContent = metaFilesNode.children?.[0].content || '';
      expect(metaFilesContent).toContain('file_011CVsVWKY9N4XdEuoaFAmJR');
      expect(metaFilesContent).toContain('document_with_chart.docx');
      // Verify it's valid JSON
      const metaFiles = JSON.parse(metaFilesContent);
      expect(metaFiles.files).toHaveLength(1);
      expect(metaFiles.files[0].filename).toBe('document_with_chart.docx');
    });
  });

  describe('tag normalization', () => {
    it('normalizes content-block-text to text element', () => {
      parser.appendChunk('<content-block-text>Hello world</content-block-text>');
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].type).toBe('element');
      expect(nodes[0].tagName).toBe('text');
      expect(nodes[0].children?.[0].content).toBe('Hello world');
    });

    it('normalizes content-block-thinking to thinking', () => {
      parser.appendChunk('<content-block-thinking>Let me think...</content-block-thinking>');
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('thinking');
      expect(nodes[0].children?.[0].content).toBe('Let me think...');
    });

    it('normalizes dynamic *_tool_result tags to server_tool_result', () => {
      // Without name attribute: extracts toolType from tag name
      parser.appendChunk(
        '<bash_code_execution_tool_result id="t1">result</bash_code_execution_tool_result>'
      );
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      // Now normalized to server_tool_result with toolType extracted from tag
      expect(nodes[0].tagName).toBe('server_tool_result');
      expect(nodes[0].attributes?.toolType).toBe('bash_code_execution');
      expect(nodes[0].attributes?.id).toBe('t1');
    });

    it('normalizes content-block-tool_call with arguments to tool_call', () => {
      // Use actual JSON (not HTML-encoded) since parseAttributes doesn't decode HTML entities
      parser.appendChunk(
        '<content-block-tool_call id="tc1" name="test_tool" arguments=\'{"key": "value"}\'></content-block-tool_call>'
      );
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('tool_call');
      expect(nodes[0].attributes?.id).toBe('tc1');
      expect(nodes[0].attributes?.name).toBe('test_tool');
      // Arguments should be parsed and formatted as JSON
      const args = JSON.parse(nodes[0].children?.[0].content || '{}');
      expect(args.key).toBe('value');
    });
  });

  describe('CDATA handling', () => {
    it('preserves CDATA content in tool results', () => {
      parser.appendChunk(
        '<content-block-tool_result id="r1" name="tool_result"><![CDATA[{"status": "success", "data": [1, 2, 3]}]]></content-block-tool_result>'
      );
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('tool_result');
      const content = nodes[0].children?.[0].content || '';
      expect(content).toBe('{"status": "success", "data": [1, 2, 3]}');
    });

    it('handles CDATA with special characters', () => {
      parser.appendChunk(
        '<content-block-tool_result id="r1" name="tool_result"><![CDATA[Line 1\nLine 2\n<tag>not xml</tag>]]></content-block-tool_result>'
      );
      const nodes = parser.getNodes();
      const content = nodes[0].children?.[0].content || '';
      expect(content).toContain('Line 1\nLine 2');
      expect(content).toContain('<tag>not xml</tag>');
    });

    it('handles multiple CDATA sections', () => {
      parser.appendChunk(
        '<content-block-tool_result id="r1" name="test"><![CDATA[first]]> and <![CDATA[second]]></content-block-tool_result>'
      );
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('tool_result');
      // Content should have both CDATA sections restored
      const childContent = nodes[0].children?.map(c => c.content).join('') || '';
      expect(childContent).toContain('first');
      expect(childContent).toContain('second');
    });
  });

  describe('streaming behavior', () => {
    it('handles incremental chunk appending', () => {
      parser.appendChunk('<content-block-');
      expect(parser.getNodes().length).toBeGreaterThanOrEqual(0); // May be 0 or partial

      parser.appendChunk('text>Hello');
      let nodes = parser.getNodes();
      // Should now have partial text content
      expect(nodes.length).toBeGreaterThanOrEqual(0);

      parser.appendChunk(' World</content-block-text>');
      nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('text');
      expect(nodes[0].children?.[0].content).toBe('Hello World');
    });

    it('allows real-time getNodes() calls', () => {
      // Append partial content
      parser.appendChunk('<content-block-thinking>Thinking about');
      
      // Should be able to get partial state without crashing
      const partialNodes = parser.getNodes();
      expect(partialNodes.length).toBeGreaterThanOrEqual(0);

      // Complete the content
      parser.appendChunk(' the problem</content-block-thinking>');
      const finalNodes = parser.getNodes();
      expect(finalNodes.length).toBe(1);
      expect(finalNodes[0].children?.[0].content).toBe('Thinking about the problem');
    });

    it('maintains buffer across multiple appendChunk calls', () => {
      const chunks = [
        '<meta_init data="test">',
        '</meta_init>',
        '<content-block-text>',
        'Content here',
        '</content-block-text>',
      ];

      chunks.forEach(chunk => parser.appendChunk(chunk));

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(2);
      expect(nodes[0].tagName).toBe('meta_init');
      expect(nodes[1].tagName).toBe('text');
      expect(nodes[1].children?.[0].content).toBe('Content here');
    });
  });

  describe('edge cases', () => {
    it('returns empty array for empty buffer', () => {
      const nodes = parser.getNodes();
      expect(nodes).toEqual([]);
    });

    it('handles reset correctly', () => {
      parser.appendChunk('<content-block-text>Some content</content-block-text>');
      expect(parser.getNodes().length).toBe(1);

      parser.reset();
      expect(parser.getNodes()).toEqual([]);
      expect(parser.getRawContent()).toBe('');
    });

    it('handles empty tags', () => {
      parser.appendChunk('<content-block-text></content-block-text>');
      const nodes = parser.getNodes();
      // Empty text block might result in empty array or single node
      expect(nodes.length).toBeLessThanOrEqual(1);
    });

    it('handles self-closing tags', () => {
      parser.appendChunk('<meta_init data="test" />');
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('meta_init');
    });

    it('treats unrecognized tags as plain text', () => {
      // Unrecognized tags like <outer>, <inner> are not parsed as XML elements
      // They are preserved as plain text (for GFM/markdown rendering)
      parser.appendChunk('<outer><inner>nested content</inner></outer>');
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      expect(nodes[0].type).toBe('text');
      expect(nodes[0].content).toBe('<outer><inner>nested content</inner></outer>');
    });

    it('handles unclosed tags gracefully', () => {
      parser.appendChunk('<content-block-text>Incomplete content');
      // Should not crash
      const nodes = parser.getNodes();
      expect(nodes.length).toBeGreaterThanOrEqual(0);
    });

    it('handles whitespace-only content', () => {
      parser.appendChunk('   \n\n   ');
      const nodes = parser.getNodes();
      // Whitespace-only should result in empty or whitespace text node
      expect(nodes.length).toBeLessThanOrEqual(1);
    });
  });

  describe('getRawContent', () => {
    it('returns accumulated buffer content', () => {
      parser.appendChunk('first ');
      parser.appendChunk('second ');
      parser.appendChunk('third');

      expect(parser.getRawContent()).toBe('first second third');
    });
  });

  describe('tool_call argument parsing', () => {
    it('parses JSON arguments correctly', () => {
      // Use single quotes for attribute value to allow double quotes in JSON
      parser.appendChunk(
        `<content-block-tool_call id="tc1" name="text_editor" arguments='{"command": "create", "path": "/tmp/test.py"}'></content-block-tool_call>`
      );

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
      
      const parsedArgs = JSON.parse(nodes[0].children?.[0].content || '{}');
      expect(parsedArgs.command).toBe('create');
      expect(parsedArgs.path).toBe('/tmp/test.py');
    });

    it('handles malformed JSON arguments gracefully', () => {
      parser.appendChunk(
        '<content-block-tool_call id="tc1" name="test" arguments="{invalid json}"></content-block-tool_call>'
      );

      // Should not crash
      const nodes = parser.getNodes();
      expect(nodes.length).toBe(1);
    });
  });

  describe('mixed content', () => {
    it('handles multiple block types in sequence', () => {
      // No whitespace between tags to avoid text nodes
      const mixedContent = 
        '<content-block-thinking>Thinking...</content-block-thinking>' +
        '<content-block-text>Response text</content-block-text>' +
        '<content-block-tool_call id="t1" name="tool" arguments="{}"></content-block-tool_call>';
      parser.appendChunk(mixedContent);

      const nodes = parser.getNodes();
      expect(nodes.length).toBe(3);
      expect(nodes[0].tagName).toBe('thinking');
      expect(nodes[1].tagName).toBe('text');
      expect(nodes[1].children?.[0].content).toBe('Response text');
      expect(nodes[2].tagName).toBe('tool_call');
    });
  });

  describe('fixture-based tests', () => {
    /**
     * Helper to process SSE stream data through the XML parser.
     * Extracts XML content from SSE lines and feeds them to the parser.
     */
    function processFixtureStream(stream: string): void {
      const dataLines = extractSseData(stream);
      for (const data of dataLines) {
        if (data === '[DONE]') continue;
        // For XML format, append the data directly (skip meta_init for content parsing)
        parser.appendChunk(data);
      }
    }

    it('parses HELLO_XML fixture correctly', () => {
      processFixtureStream(HELLO_XML.stream);
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBeGreaterThan(0);
      expect(hasExpectedNodeTypes(nodes, HELLO_XML.expectedNodeTypes)).toBe(true);
      
      // Verify text content
      const textNode = nodes.find(n => n.type === 'element' && n.tagName === 'text');
      expect(textNode).toBeDefined();
      expect(textNode?.children?.[0].content).toContain('Hello');
    });

    it('parses CLIENT_TOOLS_XML fixture correctly', () => {
      processFixtureStream(CLIENT_TOOLS_XML.stream);
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBeGreaterThan(0);
      expect(hasExpectedNodeTypes(nodes, CLIENT_TOOLS_XML.expectedNodeTypes)).toBe(true);
      
      // Verify tool_call node exists
      const toolCallNode = nodes.find(n => n.type === 'element' && n.tagName === 'tool_call');
      expect(toolCallNode).toBeDefined();
      expect(toolCallNode?.attributes?.name).toBe('divide');
    });

    it('parses SERVER_TOOLS_XML fixture correctly', () => {
      processFixtureStream(SERVER_TOOLS_XML.stream);
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBeGreaterThan(0);
      expect(hasExpectedNodeTypes(nodes, SERVER_TOOLS_XML.expectedNodeTypes)).toBe(true);
      
      // Verify server_tool_call node exists
      const serverToolCallNode = nodes.find(
        n => n.type === 'element' && n.tagName === 'server_tool_call'
      );
      expect(serverToolCallNode).toBeDefined();
      expect(serverToolCallNode?.attributes?.name).toBe('web_search');
    });

    it('parses CITATIONS_XML fixture correctly', () => {
      processFixtureStream(CITATIONS_XML.stream);
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBeGreaterThan(0);
      expect(hasExpectedNodeTypes(nodes, CITATIONS_XML.expectedNodeTypes)).toBe(true);
      
      // Verify citations node exists
      const citationsNode = nodes.find(n => n.type === 'element' && n.tagName === 'citations');
      expect(citationsNode).toBeDefined();
    });
  });

  describe('server_tool_result normalization', () => {
    it('normalizes server tool result tags to server_tool_result with toolType', () => {
      // When no name attribute, extracts toolType from tag name
      parser.appendChunk(
        '<content-block-web_search_tool_result id="srvtoolu_123"><![CDATA[Search results]]></content-block-web_search_tool_result>'
      );
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('server_tool_result');
      expect(nodes[0].attributes?.toolType).toBe('web_search');
    });

    it('uses name attribute as toolType when present', () => {
      parser.appendChunk(
        '<content-block-web_search_tool_result id="srvtoolu_123" name="web_search_tool_result"><![CDATA[Search results]]></content-block-web_search_tool_result>'
      );
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('server_tool_result');
      // When name attribute is set, it's used directly as toolType
      expect(nodes[0].attributes?.toolType).toBe('web_search_tool_result');
    });

    it('preserves server_tool_call tag name', () => {
      parser.appendChunk(
        '<content-block-server_tool_call id="srvtoolu_123" name="web_search" arguments=\'{"query": "test"}\'></content-block-server_tool_call>'
      );
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('server_tool_call');
      expect(nodes[0].attributes?.name).toBe('web_search');
    });

    it('does not normalize regular tool_result to server_tool_result', () => {
      parser.appendChunk(
        '<content-block-tool_result id="toolu_123" name="divide"><![CDATA[0.5]]></content-block-tool_result>'
      );
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('tool_result');
      // Should not have toolType attribute
      expect(nodes[0].attributes?.toolType).toBeUndefined();
    });
  });

  describe('partial CDATA handling', () => {
    it('defers parsing when CDATA is incomplete', () => {
      // Append partial CDATA
      parser.appendChunk('<content-block-tool_result id="r1" name="test"><![CDATA[partial content');
      
      // Should not crash, may return empty or partial nodes
      const partialNodes = parser.getNodes();
      expect(partialNodes.length).toBeGreaterThanOrEqual(0);
      
      // Complete the CDATA
      parser.appendChunk(' more content]]></content-block-tool_result>');
      const finalNodes = parser.getNodes();
      
      expect(finalNodes.length).toBe(1);
      expect(finalNodes[0].tagName).toBe('tool_result');
      expect(finalNodes[0].children?.[0].content).toContain('partial content');
      expect(finalNodes[0].children?.[0].content).toContain('more content');
    });
  });

  describe('HTML entity decoding', () => {
    it('decodes HTML entities in attribute values', () => {
      parser.appendChunk(
        '<content-block-tool_call id="tc1" name="test" arguments="{&quot;key&quot;: &quot;value&quot;}"></content-block-tool_call>'
      );
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].tagName).toBe('tool_call');
      
      // Arguments should be decoded and parseable as JSON
      const content = nodes[0].children?.[0].content || '';
      const args = JSON.parse(content);
      expect(args.key).toBe('value');
    });

    it('decodes &amp; in attributes', () => {
      parser.appendChunk('<meta_init data="foo&amp;bar"></meta_init>');
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].attributes?.data).toBe('foo&bar');
    });

    it('decodes &lt; and &gt; in attributes', () => {
      parser.appendChunk('<meta_init data="a &lt; b &gt; c"></meta_init>');
      const nodes = parser.getNodes();
      
      expect(nodes.length).toBe(1);
      expect(nodes[0].attributes?.data).toBe('a < b > c');
    });
  });
});
