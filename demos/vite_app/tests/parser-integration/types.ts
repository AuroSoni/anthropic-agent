/**
 * Type definitions for parser integration tests.
 */
import type { AgentNode } from '../../src/lib/parsers';

/**
 * Agent types supported by the FastAPI backend (JSON format only).
 */
export type AgentType =
  | 'agent_no_tools'
  | 'agent_client_tools'
  | 'agent_all_json'
  | 'agent_frontend_tools'
  | 'agent_skills';

/**
 * Test case definition.
 */
export interface TestCase {
  /** Unique name for the test case */
  name: string;
  /** Prompt to send - can be string or complex content object */
  prompt: string | Record<string, unknown> | Array<Record<string, unknown>>;
  /** Agent type to use */
  agentType: AgentType;
  /** Optional description of what the test validates */
  description?: string;
  /** Which endpoint to call: 'json' (default) = POST /agent/run, 'multipart' = POST /agent/run/multipart */
  endpoint?: 'json' | 'multipart';
  /** Files for multipart upload: each has filename, mimeType, and a source (url or inline content) */
  files?: Array<{
    filename: string;
    mimeType: string;
    /** HTTP URL or local file path to fetch the file from */
    url?: string;
    /** Inline base64-encoded content (for small text fixtures like CSV) */
    content?: string;
  }>;
  /** If set, the test expects an HTTP error with this status code instead of a successful SSE stream */
  expectError?: number;
}

/**
 * Result from running a test case.
 */
export interface TestResult {
  /** Name of the test case */
  testCase: string;
  /** Agent type used */
  agentType: string;
  /** ISO timestamp when test was run */
  timestamp: string;
  /** Stream format (always json) */
  streamFormat: 'json';
  /** Parsed node tree */
  nodes: AgentNode[];
  /** Number of SSE chunks received */
  rawChunksCount: number;
  /** Total time to complete parsing (ms) */
  parseTimeMs: number;
  /** Error message if test failed */
  error?: string;
  /** HTTP status code (for expected-error tests) */
  httpStatus?: number;
}
