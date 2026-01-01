/**
 * Type definitions for parser integration tests.
 */
import type { AgentNode } from '../../src/lib/parsers';

/**
 * Agent types supported by the FastAPI backend.
 */
export type AgentType = 
  | 'agent_no_tools' 
  | 'agent_client_tools' 
  | 'agent_all_raw' 
  | 'agent_all_xml';

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
  /** Detected stream format (xml or raw) */
  streamFormat: string;
  /** Parsed node tree */
  nodes: AgentNode[];
  /** Number of SSE chunks received */
  rawChunksCount: number;
  /** Total time to complete parsing (ms) */
  parseTimeMs: number;
  /** Error message if test failed */
  error?: string;
}

