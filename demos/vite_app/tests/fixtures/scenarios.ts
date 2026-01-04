/**
 * Test scenarios with SSE streams derived from real API logs (temp_logs).
 * Each scenario includes the raw SSE stream and expected node types.
 */

import type { Scenario } from './helpers';

/**
 * Simple hello response in XML format
 * Source: temp_logs/hello_agent_all_xml_*.log
 */
export const HELLO_XML: Scenario = {
  name: 'hello_xml',
  format: 'xml',
  stream: `data: <meta_init data="{&quot;format&quot;: &quot;xml&quot;, &quot;user_query&quot;: &quot;Hello&quot;, &quot;message_history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: [{&quot;type&quot;: &quot;text&quot;, &quot;text&quot;: &quot;Hello&quot;}]}], &quot;agent_uuid&quot;: &quot;200af101-ddd7-489e-b5ad-6fe5fdcd31ff&quot;, &quot;model&quot;: &quot;claude-sonnet-4-5&quot;}"></meta_init>

data: <content-block-thinking>

data: The user is just

data:  greeting

data:  me with

data:  "Hello". I

data:  should respond in

data:  a friendly and helpful

data:  manner

data: ,

data:  letting

data:  them know I'm here

data:  to assist

data:  them with whatever

data:  they need.

data: </content-block-thinking>

data: <content-block-text>

data: Hello! How

data:  can I help you today?

data: </content-block-text>

data: [DONE]
`,
  expectedNodeTypes: ['thinking', 'text'],
};

/**
 * No tools response in raw format
 * Source: temp_logs/no_tools_what_is_ai_agent_no_tools_*.log (simplified)
 */
export const NO_TOOLS_RAW: Scenario = {
  name: 'no_tools_raw',
  format: 'raw',
  stream: `data: <meta_init data="{&quot;format&quot;: &quot;raw&quot;, &quot;user_query&quot;: &quot;What is AI?&quot;, &quot;message_history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: [{&quot;type&quot;: &quot;text&quot;, &quot;text&quot;: &quot;What is AI?&quot;}]}], &quot;agent_uuid&quot;: &quot;test-uuid&quot;, &quot;model&quot;: &quot;claude-sonnet-4-5&quot;}"></meta_init>

data: {"message":{"id":"msg_test","content":[],"model":"claude-sonnet-4-5","role":"assistant","stop_reason":null,"type":"message"},"type":"message_start"}

data: {"content_block":{"signature":"","thinking":"","type":"thinking"},"index":0,"type":"content_block_start"}

data: {"delta":{"thinking":"The user is asking about AI.","type":"thinking_delta"},"index":0,"type":"content_block_delta"}

data: {"index":0,"type":"content_block_stop","content_block":{"thinking":"The user is asking about AI.","type":"thinking"}}

data: {"content_block":{"citations":null,"text":"","type":"text"},"index":1,"type":"content_block_start"}

data: {"delta":{"text":"AI stands for Artificial Intelligence.","type":"text_delta"},"index":1,"type":"content_block_delta"}

data: {"index":1,"type":"content_block_stop","content_block":{"citations":null,"text":"AI stands for Artificial Intelligence.","type":"text"}}

data: {"delta":{"stop_reason":"end_turn"},"type":"message_delta"}

data: {"type":"message_stop"}

data: [DONE]
`,
  expectedNodeTypes: ['thinking', 'text'],
};

/**
 * Client tool calls in XML format (math calculation)
 * Source: temp_logs/xml_server_tools_raw_math_expression_agent_all_xml_*.log
 */
export const CLIENT_TOOLS_XML: Scenario = {
  name: 'client_tools_xml',
  format: 'xml',
  stream: `data: <meta_init data="{&quot;format&quot;: &quot;xml&quot;, &quot;user_query&quot;: &quot;What is 5*(3+2/4)?&quot;, &quot;message_history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: [{&quot;type&quot;: &quot;text&quot;, &quot;text&quot;: &quot;What is 5*(3+2/4)?&quot;}]}], &quot;agent_uuid&quot;: &quot;test-uuid&quot;, &quot;model&quot;: &quot;claude-sonnet-4-5&quot;}"></meta_init>

data: <content-block-thinking>

data: The user wants me to calculate 5*(3+2/4).

data: </content-block-thinking>

data: <content-block-text>

data: I'll calculate this step by step.

data: </content-block-text>

data: <content-block-tool_call id="toolu_01QjifwjEbYkThuC1etXy6oT" name="divide" arguments="{&quot;a&quot;: 2, &quot;b&quot;: 4}"></content-block-tool_call>

data: <content-block-tool_call id="toolu_01TW7hzRYSMBgtbmL1bWioHa" name="add" arguments="{&quot;a&quot;: 3, &quot;b&quot;: 0.5}"></content-block-tool_call>

data: <content-block-tool_call id="toolu_01Y6KcJuUZW8HyhcSKZGcWPr" name="multiply" arguments="{&quot;a&quot;: 5, &quot;b&quot;: 3.5}"></content-block-tool_call>

data: <content-block-tool_result id="toolu_01QjifwjEbYkThuC1etXy6oT" name="divide"><![CDATA[0.5]]></content-block-tool_result>

data: <content-block-tool_result id="toolu_01TW7hzRYSMBgtbmL1bWioHa" name="add"><![CDATA[3.5]]></content-block-tool_result>

data: <content-block-tool_result id="toolu_01Y6KcJuUZW8HyhcSKZGcWPr" name="multiply"><![CDATA[17.5]]></content-block-tool_result>

data: <content-block-text>

data: **5 × (3 + 2/4) = 17.5**

data: </content-block-text>

data: [DONE]
`,
  expectedNodeTypes: ['thinking', 'text', 'tool_call', 'tool_result'],
};

/**
 * Server tool call (web search) in XML format
 * Source: temp_logs/xml_ycombinator_latest_happenings_agent_all_xml_*.log (simplified)
 */
export const SERVER_TOOLS_XML: Scenario = {
  name: 'server_tools_xml',
  format: 'xml',
  stream: `data: <meta_init data="{&quot;format&quot;: &quot;xml&quot;, &quot;user_query&quot;: &quot;Search for latest news&quot;, &quot;message_history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: [{&quot;type&quot;: &quot;text&quot;, &quot;text&quot;: &quot;Search for latest news&quot;}]}], &quot;agent_uuid&quot;: &quot;test-uuid&quot;, &quot;model&quot;: &quot;claude-sonnet-4-5&quot;}"></meta_init>

data: <content-block-thinking>

data: The user wants to search for news. I'll use web_search.

data: </content-block-thinking>

data: <content-block-server_tool_call id="srvtoolu_01DLuJB77ZEAziKHnUo3MfV2" name="web_search" arguments="{&quot;query&quot;: &quot;latest tech news 2025&quot;}"></content-block-server_tool_call>

data: <content-block-web_search_tool_result id="srvtoolu_01DLuJB77ZEAziKHnUo3MfV2" name="web_search_tool_result"><![CDATA[Search results for "latest tech news 2025"]]></content-block-web_search_tool_result>

data: <content-block-text>

data: Based on my search, here are the latest updates.

data: </content-block-text>

data: [DONE]
`,
  expectedNodeTypes: ['thinking', 'server_tool_call', 'server_tool_result', 'text'],
};

/**
 * Citations in raw format
 * Source: temp_logs/content_citations_grass_sky_agent_all_raw_*.log (simplified)
 */
export const CITATIONS_RAW: Scenario = {
  name: 'citations_raw',
  format: 'raw',
  stream: `data: <meta_init data="{&quot;format&quot;: &quot;raw&quot;, &quot;user_query&quot;: &quot;What color is grass?&quot;, &quot;message_history&quot;: [], &quot;agent_uuid&quot;: &quot;test-uuid&quot;, &quot;model&quot;: &quot;claude-sonnet-4-5&quot;}"></meta_init>

data: {"message":{"id":"msg_test","content":[],"model":"claude-sonnet-4-5","role":"assistant","type":"message"},"type":"message_start"}

data: {"content_block":{"signature":"","thinking":"","type":"thinking"},"index":0,"type":"content_block_start"}

data: {"delta":{"thinking":"The user is asking about colors.","type":"thinking_delta"},"index":0,"type":"content_block_delta"}

data: {"index":0,"type":"content_block_stop","content_block":{"thinking":"The user is asking about colors.","type":"thinking"}}

data: {"content_block":{"citations":null,"text":"","type":"text"},"index":1,"type":"content_block_start"}

data: {"delta":{"text":"Based on the document, ","type":"text_delta"},"index":1,"type":"content_block_delta"}

data: {"index":1,"type":"content_block_stop","content_block":{"citations":null,"text":"Based on the document, ","type":"text"}}

data: {"content_block":{"citations":[],"text":"","type":"text"},"index":2,"type":"content_block_start"}

data: {"delta":{"citation":{"cited_text":"The grass is green.","document_index":0,"document_title":"My Document","end_char_index":20,"start_char_index":0,"type":"char_location"},"type":"citations_delta"},"index":2,"type":"content_block_delta"}

data: {"delta":{"text":"the grass is green","type":"text_delta"},"index":2,"type":"content_block_delta"}

data: {"index":2,"type":"content_block_stop","content_block":{"citations":[{"cited_text":"The grass is green.","document_index":0,"document_title":"My Document","end_char_index":20,"start_char_index":0,"type":"char_location"}],"text":"the grass is green","type":"text"}}

data: {"delta":{"stop_reason":"end_turn"},"type":"message_delta"}

data: {"type":"message_stop"}

data: [DONE]
`,
  expectedNodeTypes: ['thinking', 'text'],
};

/**
 * Citations in XML format
 * Source: temp_logs/xml_content_citations_grass_sky_agent_all_xml_*.log (simplified)
 */
export const CITATIONS_XML: Scenario = {
  name: 'citations_xml',
  format: 'xml',
  stream: `data: <meta_init data="{&quot;format&quot;: &quot;xml&quot;, &quot;user_query&quot;: &quot;What color is grass?&quot;, &quot;message_history&quot;: [], &quot;agent_uuid&quot;: &quot;test-uuid&quot;, &quot;model&quot;: &quot;claude-sonnet-4-5&quot;}"></meta_init>

data: <content-block-thinking>

data: Looking at the document for color info.

data: </content-block-thinking>

data: <content-block-text>

data: Based on the document, 

data: </content-block-text>

data: <content-block-text>

data: the grass is green

data: </content-block-text>

data: <citations><citation type="char_location" document_index="0" document_title="My Document" start_char_index="0" end_char_index="20">The grass is green.</citation></citations>

data: <content-block-text>

data: .

data: </content-block-text>

data: [DONE]
`,
  expectedNodeTypes: ['thinking', 'text', 'citations'],
};

/**
 * Client tool call in raw format
 * Source: temp_logs/client_tools_math_expression_agent_client_tools_*.log (simplified)
 * Note: This fixture tests just the raw JSON events, without the XML tool_result.
 * The actual API stream is mixed (XML meta_init/tool_result with JSON events).
 */
export const CLIENT_TOOLS_RAW: Scenario = {
  name: 'client_tools_raw',
  format: 'raw',
  stream: `data: <meta_init data="{&quot;format&quot;: &quot;raw&quot;, &quot;user_query&quot;: &quot;What is 2+2?&quot;, &quot;message_history&quot;: [], &quot;agent_uuid&quot;: &quot;test-uuid&quot;, &quot;model&quot;: &quot;claude-sonnet-4-5&quot;}"></meta_init>

data: {"message":{"id":"msg_test","content":[],"model":"claude-sonnet-4-5","role":"assistant","type":"message"},"type":"message_start"}

data: {"content_block":{"citations":null,"text":"","type":"text"},"index":0,"type":"content_block_start"}

data: {"delta":{"text":"Let me calculate that.","type":"text_delta"},"index":0,"type":"content_block_delta"}

data: {"index":0,"type":"content_block_stop","content_block":{"text":"Let me calculate that.","type":"text"}}

data: {"content_block":{"id":"toolu_01test","input":{},"name":"add","type":"tool_use"},"index":1,"type":"content_block_start"}

data: {"delta":{"partial_json":"{\\"a\\": 2, \\"b\\": 2}","type":"input_json_delta"},"index":1,"type":"content_block_delta"}

data: {"index":1,"type":"content_block_stop","content_block":{"id":"toolu_01test","input":{"a":2,"b":2},"name":"add","type":"tool_use"}}

data: {"delta":{"stop_reason":"tool_use"},"type":"message_delta"}

data: {"type":"message_stop"}

data: [DONE]
`,
  // Note: tool_result is not included as it's XML-formatted in real streams
  expectedNodeTypes: ['text', 'tool_call'],
};

/**
 * All scenarios registry
 */
export const ALL_SCENARIOS: Record<string, Scenario> = {
  hello_xml: HELLO_XML,
  no_tools_raw: NO_TOOLS_RAW,
  client_tools_xml: CLIENT_TOOLS_XML,
  client_tools_raw: CLIENT_TOOLS_RAW,
  server_tools_xml: SERVER_TOOLS_XML,
  citations_raw: CITATIONS_RAW,
  citations_xml: CITATIONS_XML,
};

/**
 * Get a scenario by name
 */
export function getScenario(name: string): Scenario {
  const scenario = ALL_SCENARIOS[name];
  if (!scenario) {
    throw new Error(`Unknown scenario: ${name}. Available: ${Object.keys(ALL_SCENARIOS).join(', ')}`);
  }
  return scenario;
}

