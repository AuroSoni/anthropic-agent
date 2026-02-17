#!/usr/bin/env npx tsx
/**
 * Parser Integration Test Runner (JSON format only)
 *
 * Calls the FastAPI agent API, parses SSE responses using JsonStreamParser,
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
  JsonStreamParser,
  parseJsonMetaInit,
  type AgentNode,
  type JsonEnvelope,
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
// File helpers
// =============================================================================

/**
 * Resolve a file source to a Buffer.
 * Supports HTTP(S) URLs, local absolute paths, and paths relative to this directory.
 */
async function fetchFileBuffer(source: string): Promise<Buffer> {
  if (source.startsWith('http://') || source.startsWith('https://')) {
    const resp = await fetch(source);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${source}: ${resp.status} ${resp.statusText}`);
    }
    return Buffer.from(await resp.arrayBuffer());
  }

  // Local file path (absolute or relative to this file's directory)
  const resolved = path.isAbsolute(source)
    ? source
    : path.resolve(__dirname, source);
  return fs.promises.readFile(resolved);
}

// =============================================================================
// Test Runner
// =============================================================================

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
  agentType?: string,
): Promise<{ nodes: AgentNode[]; chunkCount: number }> {
  const toolResults = tools.map((tool) => ({
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
      agent_type: agentType,
    }),
  });

  if (!response.ok) {
    throw new Error(`Tool results HTTP error: ${response.status} ${response.statusText}`);
  }

  const parser = new JsonStreamParser();
  let chunkCount = 0;

  for await (const data of parseSSEStream(response)) {
    chunkCount++;
    if (data === '[DONE]') continue;

    try {
      const envelope = JSON.parse(data) as JsonEnvelope;
      if (envelope.type === 'meta_init' || envelope.type === 'meta_final') continue;
      parser.processEnvelope(envelope);
    } catch {
      // Ignore parse errors in continuation
    }
  }

  return { nodes: parser.getNodes(), chunkCount };
}

/**
 * Build a fetch request for the given test case.
 * Resolves file URLs/paths to real bytes for multipart uploads.
 */
async function buildRequest(testCase: TestCase): Promise<{ url: string; init: RequestInit }> {
  if (testCase.endpoint === 'multipart') {
    const formData = new FormData();
    formData.append('user_prompt', testCase.prompt as string);
    formData.append('agent_type', testCase.agentType);

    if (testCase.files) {
      for (const file of testCase.files) {
        let bytes: Buffer;
        if (file.url) {
          bytes = await fetchFileBuffer(file.url);
        } else if (file.content) {
          bytes = Buffer.from(file.content, 'base64');
        } else {
          throw new Error(`File "${file.filename}" has neither url nor content`);
        }
        const blob = new Blob([new Uint8Array(bytes)], { type: file.mimeType });
        formData.append('files', blob, file.filename);
      }
    }

    return {
      url: `${BASE_URL}/agent/run/multipart`,
      init: { method: 'POST', body: formData },
    };
  }

  return {
    url: `${BASE_URL}/agent/run`,
    init: {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_prompt: testCase.prompt,
        agent_type: testCase.agentType,
      }),
    },
  };
}

/**
 * Run a single test case against the API.
 */
async function runTest(testCase: TestCase): Promise<TestResult> {
  const parser = new JsonStreamParser();
  let metaInitProcessed = false;
  let chunkCount = 0;
  const startTime = Date.now();
  let error: string | undefined;
  let agentUuid: string | null = null;

  try {
    const { url, init } = await buildRequest(testCase);
    const response = await fetch(url, init);

    // Handle expected-error test cases
    if (testCase.expectError) {
      if (response.status === testCase.expectError) {
        return {
          testCase: testCase.name,
          agentType: testCase.agentType,
          timestamp: new Date().toISOString(),
          streamFormat: 'json',
          nodes: [],
          rawChunksCount: 0,
          parseTimeMs: Date.now() - startTime,
          httpStatus: response.status,
        };
      } else {
        throw new Error(
          `Expected HTTP ${testCase.expectError} but got ${response.status}`,
        );
      }
    }

    if (!response.ok) {
      throw new Error(`HTTP error: ${response.status} ${response.statusText}`);
    }

    // Parse SSE stream — every data line is a JSON envelope
    for await (const data of parseSSEStream(response)) {
      chunkCount++;
      if (data === '[DONE]') continue;

      try {
        const envelope = JSON.parse(data) as JsonEnvelope;

        // Handle meta_init on first envelope
        if (!metaInitProcessed && envelope.type === 'meta_init') {
          const metaInit = parseJsonMetaInit(data);
          if (metaInit) {
            agentUuid = metaInit.agent_uuid;
          }
          metaInitProcessed = true;
          continue;
        }

        // Skip meta_final
        if (envelope.type === 'meta_final') continue;

        parser.processEnvelope(envelope);
      } catch (e) {
        const err = e instanceof Error ? e.message : String(e);
        console.warn(`  [WARN] JSON parse failed: ${err}`);
        console.warn(
          `  [WARN] Chunk (${data.length} chars): ${data.slice(0, 150)}${data.length > 150 ? '...' : ''}`,
        );
      }
    }
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
    console.error(`  [ERROR] ${error}`);
  }

  let allNodes = parser.getNodes();

  // Handle frontend tools continuation
  const pendingTools = findAwaitingFrontendTools(allNodes);
  if (pendingTools && pendingTools.length > 0 && agentUuid) {
    console.log(
      `  [INFO] Found ${pendingTools.length} frontend tool(s), submitting results...`,
    );

    try {
      const continuation = await streamToolResultsContinuation(
        agentUuid,
        pendingTools,
        testCase.agentType,
      );
      chunkCount += continuation.chunkCount;
      allNodes = [...allNodes, ...continuation.nodes];
      console.log(
        `  [INFO] Continuation complete, +${continuation.chunkCount} chunks, +${continuation.nodes.length} nodes`,
      );
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
    streamFormat: 'json',
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
      summary.push(
        `<${node.tagName}${attrs ? ` [${attrs}]` : ''}>(${childCount} children)`,
      );
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
      const flags = [
        tc.endpoint === 'multipart' ? 'multipart' : '',
        tc.expectError ? `expect ${tc.expectError}` : '',
      ]
        .filter(Boolean)
        .join(', ');
      console.log(
        `  ${tc.name.padEnd(35)} [${tc.agentType}]${flags ? ` (${flags})` : ''}`,
      );
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
  console.log(`Parser Integration Tests (JSON format)`);
  console.log(`${'='.repeat(60)}`);
  console.log(`API: ${BASE_URL}`);
  console.log(`Tests: ${testCases.length}`);
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log(`${'='.repeat(60)}\n`);

  const results: { name: string; success: boolean; file?: string; error?: string }[] =
    [];

  for (const testCase of testCases) {
    console.log(
      `[${testCases.indexOf(testCase) + 1}/${testCases.length}] Running: ${testCase.name}`,
    );
    console.log(`  Agent: ${testCase.agentType}`);
    if (testCase.endpoint === 'multipart') {
      console.log(`  Endpoint: /agent/run/multipart`);
    }
    if (testCase.expectError) {
      console.log(`  Expects: HTTP ${testCase.expectError}`);
    }

    try {
      const result = await runTest(testCase);
      const filepath = writeResult(result);

      console.log(`  Format: ${result.streamFormat}`);
      console.log(`  Chunks: ${result.rawChunksCount}`);
      console.log(`  Time: ${result.parseTimeMs}ms`);
      if (result.httpStatus) {
        console.log(`  HTTP Status: ${result.httpStatus}`);
      }
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

  const passed = results.filter((r) => r.success).length;
  const failed = results.filter((r) => !r.success).length;

  console.log(`Passed: ${passed}/${results.length}`);
  console.log(`Failed: ${failed}/${results.length}`);

  if (failed > 0) {
    console.log(`\nFailed tests:`);
    for (const r of results.filter((r) => !r.success)) {
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
