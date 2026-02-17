/**
 * Test fixtures with JSON envelope sequences for JsonStreamParser tests.
 * Derived from actual backend log files in demos/fastapi_server/temp_logs/.
 */

import type { JsonEnvelope } from '../../src/lib/parsers/json-types';

const AGENT_UUID = '11111111-1111-1111-1111-111111111111';

// Helper to create envelopes concisely
function env(
  type: string,
  delta: string,
  final: boolean,
  extras: Partial<JsonEnvelope> = {},
): JsonEnvelope {
  return { type, agent: AGENT_UUID, final, delta, ...extras };
}

/**
 * Simple text response with thinking — no tools.
 * Source: temp_logs/server_tools_raw_math_expression_agent_all_json_*.log
 */
export const SIMPLE_TEXT: JsonEnvelope[] = [
  env('meta_init', JSON.stringify({ format: 'json', user_query: 'Hi', agent_uuid: AGENT_UUID, model: 'claude-sonnet-4-5', message_history: [] }), true),
  env('thinking', 'Let me think', false),
  env('thinking', ' about this.', false),
  env('thinking', '', true),
  env('text', 'Hello! ', false),
  env('text', 'How can I help?', false),
  env('text', '', true),
  env('meta_final', JSON.stringify({ conversation_history: [], stop_reason: 'end_turn', total_steps: 1 }), true),
];

/**
 * Tool call and result sequence (client tools — calculator).
 * Source: temp_logs/client_tools_math_expression_agent_client_tools_*.log
 */
export const TOOL_CALL: JsonEnvelope[] = [
  env('thinking', 'I need to use the calculator.', false),
  env('thinking', '', true),
  env('tool_call', '{"expression": "5 * (3 + 2/4)"}', true, { id: 'tool_001', name: 'calculator' }),
  env('tool_result', '17.5', true, { id: 'tool_001', name: 'calculator' }),
  env('text', 'The answer is **17.5**.', false),
  env('text', '', true),
];

/**
 * Chunked tool call — delta accumulated over multiple envelopes.
 */
export const CHUNKED_TOOL_CALL: JsonEnvelope[] = [
  env('tool_call', '{"express', false, { id: 'tool_002', name: 'calculator' }),
  env('tool_call', 'ion": "1+1"}', true, { id: 'tool_002', name: 'calculator' }),
  env('tool_result', '2', true, { id: 'tool_002', name: 'calculator' }),
];

/**
 * Server tool call and result (web search).
 * Source: temp_logs/ycombinator_latest_happenings_agent_all_json_*.log
 */
export const SERVER_TOOL: JsonEnvelope[] = [
  env('thinking', 'I should search the web.', false),
  env('thinking', '', true),
  env('server_tool_call', '{"query": "latest YC news"}', true, { id: 'srvtool_001', name: 'web_search' }),
  env('server_tool_result', '[{"title":"YC News","url":"https://news.ycombinator.com"}]', true, { id: 'srvtool_001', name: 'web_search' }),
  env('text', 'Here are the latest results.', false),
  env('text', '', true),
];

/**
 * Citations after text block.
 * Source: temp_logs/content_citations_grass_sky_agent_all_json_*.log
 */
export const CITATIONS: JsonEnvelope[] = [
  env('text', 'The sky is blue because of Rayleigh scattering.', false),
  env('text', '', true),
  env('citation', 'Rayleigh scattering', false, {
    citation_type: 'webpage',
    document_index: '0',
    document_title: 'Why is the sky blue?',
    url: 'https://example.com/sky',
    title: 'Sky Science',
  }),
  env('citation', 'blue light wavelength', true, {
    citation_type: 'webpage',
    document_index: '0',
    document_title: 'Why is the sky blue?',
    url: 'https://example.com/sky',
    title: 'Sky Science',
  }),
];

/**
 * Tool result with image.
 * Source: temp_logs/image_receipt_usd_to_inr_agent_all_json_*.log
 */
export const TOOL_RESULT_IMAGE: JsonEnvelope[] = [
  env('tool_call', '{"path": "/tmp/receipt.png"}', true, { id: 'tool_003', name: 'read_file' }),
  env('tool_result', 'Image file contents:', false, { id: 'tool_003', name: 'read_file' }),
  env('tool_result_image', '', true, { id: 'tool_003', name: 'read_file', src: 'data:image/png;base64,iVBORw0KGgo=', media_type: 'image/png' }),
  env('tool_result', '', true, { id: 'tool_003', name: 'read_file' }),
];

/**
 * Awaiting frontend tools (user_confirm).
 * Source: temp_logs/frontend_tools_step1_pause_agent_frontend_tools_*.log
 */
export const AWAITING_FRONTEND_TOOLS: JsonEnvelope[] = [
  env('thinking', 'I need user confirmation.', false),
  env('thinking', '', true),
  env('tool_call', '{"message": "Do you want to proceed?"}', true, { id: 'tool_004', name: 'user_confirm' }),
  env('awaiting_frontend_tools', JSON.stringify([{ tool_use_id: 'tool_004', name: 'user_confirm', input: { message: 'Do you want to proceed?' } }]), true),
];

/**
 * Error envelope.
 */
export const ERROR_ENVELOPE: JsonEnvelope[] = [
  env('error', '{"message": "Rate limit exceeded", "code": 429}', true),
];

/**
 * PDF citations with page numbers.
 * Source: temp_logs/pdf_citations_model_card_key_findings_agent_all_json_*.log
 */
export const PDF_CITATIONS: JsonEnvelope[] = [
  env('text', 'According to the model card, the key findings include improved performance.', false),
  env('text', '', true),
  env('citation', 'improved benchmark performance', false, {
    citation_type: 'page_location',
    document_index: '0',
    document_title: 'Model Card',
    start_page_number: '3',
    end_page_number: '4',
  }),
  env('citation', 'reduced hallucination rates', true, {
    citation_type: 'page_location',
    document_index: '0',
    document_title: 'Model Card',
    start_page_number: '5',
    end_page_number: '5',
  }),
];

/**
 * Char-location citations (inline text document).
 * Source: temp_logs/content_citations_grass_sky_agent_all_json_*.log
 */
export const CHAR_LOCATION_CITATIONS: JsonEnvelope[] = [
  env('text', 'The grass is green and the sky is blue.', false),
  env('text', '', true),
  env('citation', 'The grass is green. ', true, {
    citation_type: 'char_location',
    document_index: '0',
    document_title: 'My Document',
    start_char_index: 0,
    end_char_index: 20,
  }),
  env('citation', 'The sky is blue.', true, {
    citation_type: 'char_location',
    document_index: '0',
    document_title: 'My Document',
    start_char_index: 20,
    end_char_index: 36,
  }),
];

/**
 * Meta files envelope (generated file download links).
 * Source: temp_logs/skills_xlsx_budget_agent_skills_*.log
 */
export const META_FILES: JsonEnvelope[] = [
  env('thinking', 'I will create the file.', false),
  env('thinking', '', true),
  env('text', 'Here is your budget spreadsheet.', false),
  env('text', '', true),
  env('meta_files', JSON.stringify({
    files: [{
      file_id: 'file_011CYE1GuHJ546z4Z8dUwqaF',
      filename: 'monthly_budget.xlsx',
      storage_location: 'https://example.com/monthly_budget.xlsx',
      size: 5841,
    }],
  }), true),
];

/**
 * Multiple text blocks (thinking → text → thinking → text pattern).
 */
export const MULTI_BLOCK: JsonEnvelope[] = [
  env('thinking', 'First thought', false),
  env('thinking', '', true),
  env('text', 'First response', false),
  env('text', '', true),
  env('thinking', 'Second thought', false),
  env('thinking', '', true),
  env('text', 'Second response', false),
  env('text', '', true),
];
