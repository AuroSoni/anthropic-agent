/**
 * Agent Stream Parser Library
 * 
 * A collection of parsers for processing Anthropic agent streaming output.
 * Supports both raw JSON events and XML-formatted streams.
 * 
 * @example
 * ```typescript
 * import { 
 *   AnthropicStreamParser, 
 *   XmlStreamParser, 
 *   parseMetaInit,
 *   type AgentNode 
 * } from './parsers';
 * 
 * // For raw format (JSON events)
 * const rawParser = new AnthropicStreamParser();
 * rawParser.processEvent(event);
 * const nodes = rawParser.getNodes();
 * 
 * // For XML format
 * const xmlParser = new XmlStreamParser();
 * xmlParser.appendChunk(chunk);
 * const nodes = xmlParser.getNodes();
 * ```
 */

// Types
export type {
  AgentNode,
  AgentNodeType,
  AnthropicEvent,
  AnthropicEventType,
  ContentBlockType,
  ContentBlockStartEvent,
  ContentBlockDeltaEvent,
  ContentBlockStopEvent,
  MessageStartEvent,
  MessageDeltaEvent,
  MessageStopEvent,
  MetaInit,
  StreamFormat,
} from './types';

// Parsers
export { AnthropicStreamParser } from './anthropic-parser';
export { XmlStreamParser } from './xml-stream-parser';
export { parseMixedContent } from './xml-parser';

// Meta-init utilities
export { parseMetaInit, stripMetaInit, hasMetaInit } from './meta-init';

