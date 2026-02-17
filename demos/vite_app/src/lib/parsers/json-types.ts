/**
 * Type definitions for the JSON envelope stream format.
 *
 * Each SSE `data:` line from the backend is a self-contained JSON object
 * with the shape: { type, agent, final, delta, ...extras }
 */

/**
 * A single JSON envelope received from the backend SSE stream.
 */
export interface JsonEnvelope {
  /** Event type: text, thinking, tool_call, tool_result, citation, etc. */
  type: string;
  /** Agent UUID that emitted this envelope */
  agent: string;
  /** Whether this is the final envelope for this content block */
  final: boolean;
  /** Payload content (text chunk, JSON string, cited text, etc.) */
  delta: string;

  // --- Extras (present depending on type) ---

  /** Tool use ID (tool_call, tool_result, tool_result_image, server_tool_call, server_tool_result) */
  id?: string;
  /** Tool name (tool_call, tool_result, tool_result_image, server_tool_call, server_tool_result) */
  name?: string;
  /** Image source data URL (tool_result_image) */
  src?: string;
  /** Image MIME type (tool_result_image) */
  media_type?: string;
  /** Citation type (citation) */
  citation_type?: string;
  /** Document index (citation) */
  document_index?: string;
  /** Document title (citation) */
  document_title?: string;
  /** Start page number (citation) */
  start_page_number?: string;
  /** End page number (citation — page_location) */
  end_page_number?: string;
  /** Start char index (citation — char_location) */
  start_char_index?: number;
  /** End char index (citation — char_location) */
  end_char_index?: number;
  /** Citation URL (citation — web) */
  url?: string;
  /** Citation title (citation) */
  title?: string;

  /** Allow additional fields for forward compatibility */
  [key: string]: unknown;
}
