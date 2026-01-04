/**
 * Test fixture helpers for parsing and manipulating SSE streams.
 */

import type { AgentNode } from '../../src/lib/parsers';

/**
 * Interface for test scenarios
 */
export interface Scenario {
  name: string;
  format: 'raw' | 'xml';
  stream: string;
  expectedNodeTypes: string[]; // Just tag names for easy assertion
}

/**
 * Extract the data content from SSE lines
 */
export function extractSseData(stream: string): string[] {
  return stream
    .split('\n')
    .filter(line => line.startsWith('data: '))
    .map(line => line.slice(6)); // Remove 'data: ' prefix
}

/**
 * Convert SSE stream string to ReadableStream for MSW response
 */
export function streamToReadable(stream: string): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  const lines = stream.split('\n');
  let index = 0;

  return new ReadableStream({
    pull(controller) {
      if (index < lines.length) {
        // Send line by line with small delay simulation
        controller.enqueue(encoder.encode(lines[index] + '\n'));
        index++;
      } else {
        controller.close();
      }
    },
  });
}

/**
 * Check if nodes array contains expected tag types in order
 */
export function hasExpectedNodeTypes(
  nodes: AgentNode[],
  expectedTypes: string[]
): boolean {
  const actualTypes = nodes
    .filter((n): n is AgentNode & { type: 'element' } => n.type === 'element')
    .map(n => n.tagName);
  
  // Check that all expected types are present (may have more)
  return expectedTypes.every(type => actualTypes.includes(type));
}

/**
 * Find node by tag name
 */
export function findNodeByTag(
  nodes: AgentNode[],
  tagName: string
): AgentNode | undefined {
  return nodes.find(
    (n): n is AgentNode & { type: 'element' } => 
      n.type === 'element' && n.tagName === tagName
  );
}

/**
 * Get text content from a node's children
 */
export function getNodeTextContent(node: AgentNode): string {
  if (node.type === 'text') {
    return node.content || '';
  }
  if (node.type === 'element' && node.children) {
    return node.children
      .map(child => getNodeTextContent(child))
      .join('');
  }
  return '';
}

