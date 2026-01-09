#!/usr/bin/env npx tsx
/**
 * Parser Integration Test Runner
 * 
 * Calls the FastAPI agent API, parses SSE responses using the parsers,
 * and logs pretty-printed JSON node trees for validation.
 * 
 * Usage:
 *   npx tsx tests/parser-integration/run-tests.ts              # Run all tests
 *   npx tsx tests/parser-integration/run-tests.ts --test math  # Run tests matching "math"
 *   npx tsx tests/parser-integration/run-tests.ts --list       # List available tests
 */
import * as fs from 'fs';
import * as path from 'path';
import { fileURLToPath } from 'url';
import {
  AnthropicStreamParser,
  XmlStreamParser,
  parseMetaInit,
  stripMetaInit,
  type AgentNode,
  type StreamFormat,
  type AnthropicEvent,
} from '../../src/lib/parsers/index.js';
import { getTestCases, TEST_CASES } from './test-cases.js';
import type { TestCase, TestResult } from './types.js';

// =============================================================================
// Configuration
// =============================================================================

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const BASE_URL = process.env.API_URL || 'http://localhost:8000';
const OUTPUT_DIR = path.join(__dirname, 'output');

// =============================================================================
// SSE Parser (Node.js compatible)
// =============================================================================

/**
 * Parse SSE stream from a fetch response.
 * Yields individual data events.
 */
async function* parseSSEStream(response: Response): AsyncGenerator<string> {
  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      
      // Process complete lines
      const lines = buffer.split('\n');
      buffer = lines.pop() || ''; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          yield line.slice(6); // Remove "data: " prefix
        }
        // Skip empty lines and other SSE fields (event:, id:, retry:)
      }
    }

    // Process remaining buffer
    if (buffer.startsWith('data: ')) {
      yield buffer.slice(6);
    }
  } finally {
    reader.releaseLock();
  }
}

// =============================================================================
// Test Runner
// =============================================================================

type StreamParser = AnthropicStreamParser | XmlStreamParser;

/**
 * Pending frontend tool from awaiting_frontend_tools tag.
 */
interface PendingFrontendTool {
  tool_use_id: string;
  name: string;
  input: Record<string, unknown>;
}

/**
 * Find awaiting_frontend_tools node and extract pending tools.
 */
function findAwaitingFrontendTools(nodes: AgentNode[]): PendingFrontendTool[] | null {
  for (const node of nodes) {
    if (node.type === 'element' && node.tagName === 'awaiting_frontend_tools') {
      const dataAttr = node.attributes?.data;
      if (dataAttr) {
        try {
          return JSON.parse(dataAttr) as PendingFrontendTool[];
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

/**
 * Submit tool results and stream continuation response.
 */
async function streamToolResultsContinuation(
  agentUuid: string,
  tools: PendingFrontendTool[],
  existingFormat: StreamFormat,
): Promise<{ nodes: AgentNode[]; chunkCount: number }> {
  // Create tool results (respond "yes" to all user_confirm tools)
  const toolResults = tools.map(tool => ({
    tool_use_id: tool.tool_use_id,
    content: tool.name === 'user_confirm' ? 'yes' : 'ok',
    is_error: false,
  }));

  const response = await fetch(`${BASE_URL}/agent/tool_results`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      agent_uuid: agentUuid,
      tool_results: toolResults,
    }),
  });

  if (!response.ok) {
    throw new Error(`Tool results HTTP error: ${response.status} ${response.statusText}`);
  }

  // Parse continuation stream
  let parser: StreamParser;
  let xmlFallbackParser: XmlStreamParser | null = null;

  if (existingFormat === 'raw') {
    parser = new AnthropicStreamParser();
    xmlFallbackParser = new XmlStreamParser();
  } else {
    parser = new XmlStreamParser();
  }

  let chunkCount = 0;

  // Track order of XML chunks relative to main parser blocks (for interleaving)
  const xmlInsertionPoints: { afterMainBlockCount: number; xmlNodeIndex: number }[] = [];
  let mainBlockCount = 0;

  for await (const data of parseSSEStream(response)) {
    chunkCount++;

    if (data === '[DONE]') {
      continue;
    }

    if (existingFormat === 'raw') {
      const trimmed = data.trim();
      if (trimmed.startsWith('<')) {
        const prevXmlCount = xmlFallbackParser?.getNodes().length || 0;
        xmlFallbackParser?.appendChunk(data);
        const newXmlCount = xmlFallbackParser?.getNodes().length || 0;
        
        // Track where each new XML node should be inserted
        for (let i = prevXmlCount; i < newXmlCount; i++) {
          xmlInsertionPoints.push({ afterMainBlockCount: mainBlockCount, xmlNodeIndex: i });
        }
      } else {
        const unescaped = data.replace(/\\\\/g, '\\');
        try {
          const event = JSON.parse(unescaped) as AnthropicEvent;
          
          // Track when new blocks start
          if (event.type === 'content_block_start') {
            mainBlockCount++;
          }
          
          (parser as AnthropicStreamParser).processEvent(event);
        } catch {
          // Ignore parse errors in continuation
        }
      }
    } else {
      (parser as XmlStreamParser).appendChunk(data);
    }
  }

  // Return interleaved nodes for raw mode, direct for xml mode
  if (existingFormat === 'raw') {
    const mainNodes = parser.getNodes();
    const xmlNodes = xmlFallbackParser?.getNodes() || [];
    const allNodes: AgentNode[] = [];
    let xmlInsertIdx = 0;
    
    for (let mainIdx = 0; mainIdx <= mainNodes.length; mainIdx++) {
      // Insert XML nodes at this position
      while (xmlInsertIdx < xmlInsertionPoints.length && 
             xmlInsertionPoints[xmlInsertIdx].afterMainBlockCount === mainIdx) {
        const xmlNodeIdx = xmlInsertionPoints[xmlInsertIdx].xmlNodeIndex;
        if (xmlNodeIdx < xmlNodes.length) {
          allNodes.push(xmlNodes[xmlNodeIdx]);
        }
        xmlInsertIdx++;
      }
      if (mainIdx < mainNodes.length) {
        allNodes.push(mainNodes[mainIdx]);
      }
    }
    // Add remaining XML nodes
    while (xmlInsertIdx < xmlInsertionPoints.length) {
      const xmlNodeIdx = xmlInsertionPoints[xmlInsertIdx].xmlNodeIndex;
      if (xmlNodeIdx < xmlNodes.length) {
        allNodes.push(xmlNodes[xmlNodeIdx]);
      }
      xmlInsertIdx++;
    }
    
    return { nodes: allNodes, chunkCount };
  } else {
    return { nodes: parser.getNodes(), chunkCount };
  }
}

/**
 * Run a single test case against the API.
 */
async function runTest(testCase: TestCase): Promise<TestResult> {
  let parser: StreamParser | null = null;
  let xmlFallbackParser: XmlStreamParser | null = null; // For XML tool results in raw mode
  let streamFormat: StreamFormat | null = null;
  let metaInitProcessed = false;
  let chunkCount = 0;
  const startTime = Date.now();
  let error: string | undefined;
  let agentUuid: string | null = null;

  // Track order of XML chunks relative to main parser blocks (for interleaving)
  // Each entry: { afterMainBlockCount: number, xmlNodeIndex: number }
  const xmlInsertionPoints: { afterMainBlockCount: number; xmlNodeIndex: number }[] = [];
  let mainBlockCount = 0;  // Tracks number of content_block_start events processed

  /**
   * Process a chunk through the appropriate parser.
   * Handles hybrid content in raw mode: JSON events + XML tool results.
   */
  function processChunk(chunk: string): void {
    if (!parser || !chunk.trim()) return;

    if (streamFormat === 'raw') {
      const trimmed = chunk.trim();
      
      // XML content (tool results) injected by backend
      if (trimmed.startsWith('<')) {
        const prevXmlCount = xmlFallbackParser?.getNodes().length || 0;
        xmlFallbackParser?.appendChunk(chunk);
        const newXmlCount = xmlFallbackParser?.getNodes().length || 0;
        
        // Track where each new XML node should be inserted (relative to main block count)
        for (let i = prevXmlCount; i < newXmlCount; i++) {
          xmlInsertionPoints.push({ afterMainBlockCount: mainBlockCount, xmlNodeIndex: i });
        }
        return;
      }
      
      // JSON events from Anthropic API
      // Backend escapes backslashes for SSE transport, so unescape before parsing
      const unescaped = chunk.replace(/\\\\/g, '\\');
      try {
        const event = JSON.parse(unescaped) as AnthropicEvent;
        
        // Track when new blocks start (for interleaving with XML nodes)
        if (event.type === 'content_block_start') {
          mainBlockCount++;
        }
        
        (parser as AnthropicStreamParser).processEvent(event);
      } catch (e) {
        // Debug: show error and full chunk
        const err = e instanceof Error ? e.message : String(e);
        console.warn(`  [WARN] JSON parse failed: ${err}`);
        console.warn(`  [WARN] Chunk (${chunk.length} chars): ${chunk.slice(0, 150)}${chunk.length > 150 ? '...' : ''}`);
      }
    } else {
      (parser as XmlStreamParser).appendChunk(chunk);
    }
  }

  try {
    const response = await fetch(`${BASE_URL}/agent/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_prompt: testCase.prompt,
        agent_type: testCase.agentType,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
    }

    for await (const data of parseSSEStream(response)) {
      chunkCount++;

      if (data === '[DONE]') {
        continue;
      }

      // Detect format from meta_init on first chunk
      if (!metaInitProcessed) {
        // Don't unescape before parseMetaInit - the JSON uses standard escaping
        // that JSON.parse() handles correctly. Unescaping would break newlines in strings.
        const metaInit = parseMetaInit(data);

        if (metaInit) {
          streamFormat = metaInit.format;
          agentUuid = metaInit.agent_uuid;
          if (streamFormat === 'raw') {
            parser = new AnthropicStreamParser();
            xmlFallbackParser = new XmlStreamParser(); // For XML tool results
          } else {
            parser = new XmlStreamParser();
          }
          metaInitProcessed = true;

          // Process remaining content after meta_init
          const remaining = stripMetaInit(data);
          if (remaining.trim()) {
            processChunk(remaining);
          }
          continue;
        }

        // Fallback format detection if no meta_init
        if (data.trim().startsWith('{')) {
          parser = new AnthropicStreamParser();
          xmlFallbackParser = new XmlStreamParser(); // For XML tool results
          streamFormat = 'raw';
        } else {
          parser = new XmlStreamParser();
          streamFormat = 'xml';
        }
        metaInitProcessed = true;
      }

      processChunk(data);
    }
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
    console.error(`  [ERROR] ${error}`);
  }

  // Collect final nodes - interleave main and XML nodes for raw mode
  let allNodes: AgentNode[];
  if (streamFormat === 'raw') {
    // Get complete nodes from both parsers (now that all deltas are processed)
    const mainNodes = parser?.getNodes() || [];
    const xmlNodes = xmlFallbackParser?.getNodes() || [];
    
    // Interleave XML nodes at their recorded insertion points
    allNodes = [];
    let xmlInsertIdx = 0;
    
    for (let mainIdx = 0; mainIdx <= mainNodes.length; mainIdx++) {
      // Insert any XML nodes that belong at this position
      while (xmlInsertIdx < xmlInsertionPoints.length && 
             xmlInsertionPoints[xmlInsertIdx].afterMainBlockCount === mainIdx) {
        const xmlNodeIdx = xmlInsertionPoints[xmlInsertIdx].xmlNodeIndex;
        if (xmlNodeIdx < xmlNodes.length) {
          allNodes.push(xmlNodes[xmlNodeIdx]);
        }
        xmlInsertIdx++;
      }
      // Add the main node
      if (mainIdx < mainNodes.length) {
        allNodes.push(mainNodes[mainIdx]);
      }
    }
    // Add any remaining XML nodes
    while (xmlInsertIdx < xmlInsertionPoints.length) {
      const xmlNodeIdx = xmlInsertionPoints[xmlInsertIdx].xmlNodeIndex;
      if (xmlNodeIdx < xmlNodes.length) {
        allNodes.push(xmlNodes[xmlNodeIdx]);
      }
      xmlInsertIdx++;
    }
  } else {
    // For XML mode, just get nodes from the single parser
    allNodes = parser?.getNodes() || [];
  }

  // Check for awaiting_frontend_tools and continue if found
  const pendingTools = findAwaitingFrontendTools(allNodes);
  if (pendingTools && pendingTools.length > 0 && agentUuid && streamFormat) {
    console.log(`  [INFO] Found ${pendingTools.length} frontend tool(s), submitting results...`);
    
    try {
      const continuation = await streamToolResultsContinuation(agentUuid, pendingTools, streamFormat);
      chunkCount += continuation.chunkCount;
      allNodes = [...allNodes, ...continuation.nodes];
      console.log(`  [INFO] Continuation complete, +${continuation.chunkCount} chunks, +${continuation.nodes.length} nodes`);
    } catch (e) {
      const contError = e instanceof Error ? e.message : String(e);
      console.error(`  [ERROR] Continuation failed: ${contError}`);
      error = error ? `${error}; Continuation: ${contError}` : contError;
    }
  }

  return {
    testCase: testCase.name,
    agentType: testCase.agentType,
    timestamp: new Date().toISOString(),
    streamFormat: streamFormat || 'unknown',
    nodes: allNodes,
    rawChunksCount: chunkCount,
    parseTimeMs: Date.now() - startTime,
    ...(error && { error }),
  };
}

/**
 * Write test result to JSON file.
 */
function writeResult(result: TestResult): string {
  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const timestamp = Date.now();
  const filename = `${result.testCase}_${timestamp}.json`;
  const filepath = path.join(OUTPUT_DIR, filename);

  fs.writeFileSync(filepath, JSON.stringify(result, null, 2));
  return filepath;
}

/**
 * Print a summary of nodes for console output.
 */
function summarizeNodes(nodes: AgentNode[]): string {
  const summary: string[] = [];

  for (const node of nodes) {
    if (node.type === 'text') {
      const content = node.content || '';
      summary.push(`text(${content.length} chars)`);
    } else if (node.type === 'element' && node.tagName) {
      const childCount = node.children?.length || 0;
      const attrs = node.attributes ? Object.keys(node.attributes).join(',') : '';
      summary.push(`<${node.tagName}${attrs ? ` [${attrs}]` : ''}>(${childCount} children)`);
    }
  }

  return summary.join(', ') || '(empty)';
}

// =============================================================================
// CLI
// =============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  // Handle --list flag
  if (args.includes('--list')) {
    console.log('\nAvailable test cases:\n');
    for (const tc of TEST_CASES) {
      console.log(`  ${tc.name.padEnd(30)} [${tc.agentType}]`);
      if (tc.description) {
        console.log(`    ${tc.description}`);
      }
    }
    console.log(`\nTotal: ${TEST_CASES.length} test cases\n`);
    return;
  }

  // Handle --test filter
  const testIndex = args.indexOf('--test');
  const filter = testIndex !== -1 ? args[testIndex + 1] : undefined;

  // Get test cases
  const testCases = filter ? getTestCases(filter) : TEST_CASES;

  if (testCases.length === 0) {
    console.error(`No test cases found${filter ? ` matching "${filter}"` : ''}`);
    process.exit(1);
  }

  console.log(`\n${'='.repeat(60)}`);
  console.log(`Parser Integration Tests`);
  console.log(`${'='.repeat(60)}`);
  console.log(`API: ${BASE_URL}`);
  console.log(`Tests: ${testCases.length}`);
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log(`${'='.repeat(60)}\n`);

  const results: { name: string; success: boolean; file?: string; error?: string }[] = [];

  for (const testCase of testCases) {
    console.log(`[${testCases.indexOf(testCase) + 1}/${testCases.length}] Running: ${testCase.name}`);
    console.log(`  Agent: ${testCase.agentType}`);

    try {
      const result = await runTest(testCase);
      const filepath = writeResult(result);

      console.log(`  Format: ${result.streamFormat}`);
      console.log(`  Chunks: ${result.rawChunksCount}`);
      console.log(`  Time: ${result.parseTimeMs}ms`);
      console.log(`  Nodes: ${summarizeNodes(result.nodes)}`);
      console.log(`  Output: ${path.basename(filepath)}`);

      if (result.error) {
        console.log(`  ERROR: ${result.error}`);
        results.push({ name: testCase.name, success: false, error: result.error });
      } else {
        results.push({ name: testCase.name, success: true, file: filepath });
      }
    } catch (e) {
      const error = e instanceof Error ? e.message : String(e);
      console.log(`  FAILED: ${error}`);
      results.push({ name: testCase.name, success: false, error });
    }

    console.log('');
  }

  // Summary
  console.log(`${'='.repeat(60)}`);
  console.log(`Summary`);
  console.log(`${'='.repeat(60)}`);

  const passed = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;

  console.log(`Passed: ${passed}/${results.length}`);
  console.log(`Failed: ${failed}/${results.length}`);

  if (failed > 0) {
    console.log(`\nFailed tests:`);
    for (const r of results.filter(r => !r.success)) {
      console.log(`  - ${r.name}: ${r.error}`);
    }
  }

  console.log('');
  process.exit(failed > 0 ? 1 : 0);
}

// Run
main().catch((e) => {
  console.error('Fatal error:', e);
  process.exit(1);
});
