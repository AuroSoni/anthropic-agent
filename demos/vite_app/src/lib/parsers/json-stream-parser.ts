/**
 * Streaming parser for JSON envelope format: processes self-contained JSON envelopes
 * and emits AgentNode[] matching the same shape as XmlStreamParser / AnthropicStreamParser.
 *
 * Each envelope has the shape: { type, agent, final, delta, ...extras }
 *
 * This is the default parser for format: "json".
 */
import type { AgentNode } from './types';
import type { JsonEnvelope } from './json-types';
import { parseMixedContent } from './xml-parser';

// ============================================================================
// Internal block state
// ============================================================================

interface StreamBlock {
  kind: 'thinking' | 'text' | 'tool_call' | 'server_tool_call' |
        'tool_result' | 'server_tool_result' | 'awaiting_frontend_tools' | 'error' |
        'meta_files';
  content: string;
  isComplete: boolean;
  extras: Record<string, string>;
  /** Citation nodes accumulated after a text block */
  citations?: CitationData[];
  /** Image nodes accumulated inside a tool_result */
  images?: ImageData[];
}

interface CitationData {
  cited_text: string;
  attrs: Record<string, string>;
}

interface ImageData {
  src: string;
  media_type: string;
}

// ============================================================================
// Parser
// ============================================================================

/**
 * JsonStreamParser processes JSON envelope events from the agent SSE stream.
 *
 * Designed for aggressive real-time updates: getNodes() can be called after
 * every envelope to get the current tree state (even with incomplete blocks).
 */
export class JsonStreamParser {
  private blocks: StreamBlock[] = [];
  /** Index of the currently open text block (for attaching citations) */
  private currentTextBlockIndex: number = -1;
  /** Index of the currently open thinking block */
  private currentThinkingBlockIndex: number = -1;
  /** Index of the currently open tool_call / server_tool_call being chunked */
  private currentToolCallIndex: number = -1;
  /** Index of the currently open tool_result / server_tool_result being chunked */
  private currentToolResultIndex: number = -1;

  /**
   * Process a single JSON envelope from the stream.
   */
  processEnvelope(envelope: JsonEnvelope): void {
    const { type, final: isFinal, delta } = envelope;

    switch (type) {
      case 'thinking':
        this.handleStreamed('thinking', delta, isFinal);
        break;

      case 'text':
        this.handleStreamed('text', delta, isFinal);
        break;

      case 'citation':
        this.handleCitation(envelope);
        break;

      case 'tool_call':
        this.handleBuffered('tool_call', delta, isFinal, envelope);
        break;

      case 'server_tool_call':
        this.handleBuffered('server_tool_call', delta, isFinal, envelope);
        break;

      case 'tool_result':
        this.handleToolResult(delta, isFinal, envelope);
        break;

      case 'tool_result_image':
        this.handleToolResultImage(envelope);
        break;

      case 'server_tool_result':
        this.handleBuffered('server_tool_result', delta, isFinal, envelope);
        break;

      case 'awaiting_frontend_tools':
        this.blocks.push({
          kind: 'awaiting_frontend_tools',
          content: delta,
          isComplete: true,
          extras: {},
        });
        break;

      case 'error':
        this.blocks.push({
          kind: 'error',
          content: delta,
          isComplete: true,
          extras: {},
        });
        break;

      case 'meta_files':
        this.blocks.push({
          kind: 'meta_files',
          content: delta,
          isComplete: true,
          extras: {},
        });
        break;

      // meta_init and meta_final are handled upstream in agent-stream.ts
      case 'meta_init':
      case 'meta_final':
        break;

      default:
        // Unknown envelope type — ignore gracefully
        break;
    }
  }

  /**
   * Handle streamed blocks (thinking, text) where deltas append content.
   */
  private handleStreamed(
    kind: 'thinking' | 'text',
    delta: string,
    isFinal: boolean,
  ): void {
    const currentIndex = kind === 'thinking'
      ? this.currentThinkingBlockIndex
      : this.currentTextBlockIndex;

    if (currentIndex >= 0 && !this.blocks[currentIndex].isComplete) {
      // Append to existing open block
      this.blocks[currentIndex].content += delta;
      if (isFinal) {
        this.blocks[currentIndex].isComplete = true;
        if (kind === 'thinking') this.currentThinkingBlockIndex = -1;
        else this.currentTextBlockIndex = -1;
      }
    } else if (!isFinal || delta) {
      // Create new block
      const idx = this.blocks.length;
      this.blocks.push({
        kind,
        content: delta,
        isComplete: isFinal,
        extras: {},
      });
      if (!isFinal) {
        if (kind === 'thinking') this.currentThinkingBlockIndex = idx;
        else this.currentTextBlockIndex = idx;
      }
    }
  }

  /**
   * Handle citation envelopes — attach to the most recent text block.
   */
  private handleCitation(envelope: JsonEnvelope): void {
    const attrs: Record<string, string> = {};
    if (envelope.citation_type) attrs.type = String(envelope.citation_type);
    if (envelope.document_index != null) attrs.document_index = String(envelope.document_index);
    if (envelope.document_title) attrs.document_title = String(envelope.document_title);
    if (envelope.start_page_number != null) attrs.start_page_number = String(envelope.start_page_number);
    if (envelope.end_page_number != null) attrs.end_page_number = String(envelope.end_page_number);
    if (envelope.start_char_index != null) attrs.start_char_index = String(envelope.start_char_index);
    if (envelope.end_char_index != null) attrs.end_char_index = String(envelope.end_char_index);
    if (envelope.url) attrs.url = String(envelope.url);
    if (envelope.title) attrs.title = String(envelope.title);

    const citation: CitationData = {
      cited_text: envelope.delta || '',
      attrs,
    };

    // Find the last text block to attach citations to
    let targetIndex = -1;
    for (let i = this.blocks.length - 1; i >= 0; i--) {
      if (this.blocks[i].kind === 'text') {
        targetIndex = i;
        break;
      }
    }

    if (targetIndex >= 0) {
      if (!this.blocks[targetIndex].citations) {
        this.blocks[targetIndex].citations = [];
      }
      this.blocks[targetIndex].citations!.push(citation);
    } else {
      // No preceding text block — create a standalone citations holder
      this.blocks.push({
        kind: 'text',
        content: '',
        isComplete: true,
        extras: {},
        citations: [citation],
      });
    }
  }

  /**
   * Handle buffered blocks (tool_call, server_tool_call, server_tool_result)
   * where deltas accumulate until final:true.
   */
  private handleBuffered(
    kind: 'tool_call' | 'server_tool_call' | 'server_tool_result',
    delta: string,
    isFinal: boolean,
    envelope: JsonEnvelope,
  ): void {
    const isToolCall = kind === 'tool_call' || kind === 'server_tool_call';
    const currentIndex = isToolCall ? this.currentToolCallIndex : this.currentToolResultIndex;

    if (currentIndex >= 0 && !this.blocks[currentIndex].isComplete) {
      // Append to existing open block
      this.blocks[currentIndex].content += delta;
      if (isFinal) {
        this.blocks[currentIndex].isComplete = true;
        if (isToolCall) this.currentToolCallIndex = -1;
        else this.currentToolResultIndex = -1;
      }
    } else {
      const extras: Record<string, string> = {};
      if (envelope.id) extras.id = envelope.id;
      if (envelope.name) extras.name = envelope.name;

      const idx = this.blocks.length;
      this.blocks.push({
        kind,
        content: delta,
        isComplete: isFinal,
        extras,
      });
      if (!isFinal) {
        if (isToolCall) this.currentToolCallIndex = idx;
        else this.currentToolResultIndex = idx;
      }
    }
  }

  /**
   * Handle tool_result envelopes — may include interleaved tool_result_image.
   */
  private handleToolResult(
    delta: string,
    isFinal: boolean,
    envelope: JsonEnvelope,
  ): void {
    if (this.currentToolResultIndex >= 0 && !this.blocks[this.currentToolResultIndex].isComplete) {
      // Append to existing open tool_result
      this.blocks[this.currentToolResultIndex].content += delta;
      if (isFinal) {
        this.blocks[this.currentToolResultIndex].isComplete = true;
        this.currentToolResultIndex = -1;
      }
    } else {
      const extras: Record<string, string> = {};
      if (envelope.id) extras.id = envelope.id;
      if (envelope.name) extras.name = envelope.name;

      const idx = this.blocks.length;
      this.blocks.push({
        kind: 'tool_result',
        content: delta,
        isComplete: isFinal,
        extras,
      });
      if (!isFinal) {
        this.currentToolResultIndex = idx;
      }
    }
  }

  /**
   * Handle tool_result_image — attach to the current open tool_result or create standalone.
   */
  private handleToolResultImage(envelope: JsonEnvelope): void {
    const image: ImageData = {
      src: String(envelope.src || ''),
      media_type: String(envelope.media_type || ''),
    };

    if (this.currentToolResultIndex >= 0) {
      const block = this.blocks[this.currentToolResultIndex];
      if (!block.images) block.images = [];
      block.images.push(image);
    } else {
      // Find the most recent tool_result block
      for (let i = this.blocks.length - 1; i >= 0; i--) {
        if (this.blocks[i].kind === 'tool_result') {
          if (!this.blocks[i].images) this.blocks[i].images = [];
          this.blocks[i].images!.push(image);
          return;
        }
      }
      // No tool_result found — create standalone image block (shouldn't happen normally)
      this.blocks.push({
        kind: 'tool_result',
        content: '',
        isComplete: false,
        extras: {
          id: String(envelope.id || ''),
          name: String(envelope.name || ''),
        },
        images: [image],
      });
    }
  }

  // ============================================================================
  // Node generation
  // ============================================================================

  /**
   * Convert current state to AgentNode[].
   * Can be called at any time for real-time updates.
   */
  getNodes(): AgentNode[] {
    const nodes: AgentNode[] = [];

    for (const block of this.blocks) {
      switch (block.kind) {
        case 'thinking':
          nodes.push({
            type: 'element',
            tagName: 'thinking',
            children: [{ type: 'text', content: block.content }],
          });
          break;

        case 'text': {
          // Parse text content for mixed XML tags (chart, image, etc.)
          const textChildren = parseMixedContent(block.content);
          nodes.push({
            type: 'element',
            tagName: 'text',
            children: textChildren,
          });
          // Emit citations node if present
          if (block.citations && block.citations.length > 0) {
            nodes.push(this.buildCitationsNode(block.citations));
          }
          break;
        }

        case 'tool_call':
        case 'server_tool_call': {
          let content = block.content;
          if (block.isComplete) {
            try {
              const parsed = JSON.parse(block.content);
              content = JSON.stringify(parsed, null, 2);
            } catch {
              // Keep raw content if JSON parsing fails
            }
          }
          nodes.push({
            type: 'element',
            tagName: block.kind,
            attributes: {
              id: block.extras.id || '',
              name: block.extras.name || '',
            },
            children: [{ type: 'text', content }],
          });
          break;
        }

        case 'tool_result': {
          nodes.push({
            type: 'element',
            tagName: 'tool_result',
            attributes: {
              id: block.extras.id || '',
              name: block.extras.name || '',
            },
            children: [{ type: 'text', content: block.content }],
          });
          // Emit image nodes after tool result
          if (block.images) {
            for (const img of block.images) {
              nodes.push({
                type: 'element',
                tagName: 'image',
                attributes: {
                  src: img.src,
                  media_type: img.media_type,
                },
              });
            }
          }
          break;
        }

        case 'server_tool_result':
          nodes.push({
            type: 'element',
            tagName: 'server_tool_result',
            attributes: {
              id: block.extras.id || '',
              name: block.extras.name || '',
            },
            children: [{ type: 'text', content: block.content }],
          });
          break;

        case 'awaiting_frontend_tools':
          nodes.push({
            type: 'element',
            tagName: 'awaiting_frontend_tools',
            attributes: {
              data: block.content,
            },
          });
          break;

        case 'error':
          nodes.push({
            type: 'element',
            tagName: 'error',
            children: [{ type: 'text', content: block.content }],
          });
          break;

        case 'meta_files':
          nodes.push({
            type: 'element',
            tagName: 'meta_files',
            children: [{ type: 'text', content: block.content }],
          });
          break;
      }
    }

    return nodes;
  }

  /**
   * Build a citations node matching the structure expected by NodeRenderer.
   */
  private buildCitationsNode(citations: CitationData[]): AgentNode {
    return {
      type: 'element',
      tagName: 'citations',
      attributes: {},
      children: citations.map(c => ({
        type: 'element' as const,
        tagName: 'citation',
        attributes: c.attrs,
        children: [{ type: 'text' as const, content: c.cited_text }],
      })),
    };
  }

  /**
   * Reset the parser state.
   */
  reset(): void {
    this.blocks = [];
    this.currentTextBlockIndex = -1;
    this.currentThinkingBlockIndex = -1;
    this.currentToolCallIndex = -1;
    this.currentToolResultIndex = -1;
  }
}
