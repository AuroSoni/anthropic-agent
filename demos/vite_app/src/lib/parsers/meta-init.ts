/**
 * Utilities for the backend `<meta_init ...>` and `<meta_final ...>` tags: detect/parse/strip
 * initialization and final metadata before routing the stream to the correct parser.
 */
import type { MetaFinal, MetaInit } from './types';
import { decodeHtmlEntities } from './utils';

/**
 * Parse meta_init tag from content string.
 * 
 * The meta_init tag is emitted by the backend at the start of a stream
 * and contains format and metadata information.
 * 
 * Format: <meta_init data="{escaped_json}"></meta_init>
 * 
 * @param content - The content string to parse
 * @returns MetaInit object if found, null otherwise
 */
export function parseMetaInit(content: string): MetaInit | null {
  // Match <meta_init data="..."></meta_init>
  const match = content.match(/<meta_init\s+data="([^"]*)"[^>]*><\/meta_init>/);
  if (!match) return null;
  
  try {
    const escapedJson = match[1];
    const json = decodeHtmlEntities(escapedJson);
    return JSON.parse(json) as MetaInit;
  } catch (e) {
    console.warn('Failed to parse meta_init:', e);
    return null;
  }
}

/**
 * Remove meta_init tag from content string.
 * 
 * @param content - The content string to strip
 * @returns Content with meta_init tag removed
 */
export function stripMetaInit(content: string): string {
  return content.replace(/<meta_init\s+data="[^"]*"[^>]*><\/meta_init>/, '');
}

/**
 * Check if content contains a meta_init tag.
 * 
 * @param content - The content string to check
 * @returns true if meta_init tag is present
 */
export function hasMetaInit(content: string): boolean {
  return /<meta_init\s+data="[^"]*"[^>]*><\/meta_init>/.test(content);
}

/**
 * Parse meta_final tag from content string.
 * 
 * Format: <meta_final data="{escaped_json}"></meta_final>
 */
export function parseMetaFinal(content: string): MetaFinal | null {
  const match = content.match(/<meta_final\s+data="([^"]*)"[^>]*><\/meta_final>/);
  if (!match) return null;
  
  try {
    const escapedJson = match[1];
    const json = decodeHtmlEntities(escapedJson);
    return JSON.parse(json) as MetaFinal;
  } catch (e) {
    console.warn('Failed to parse meta_final:', e);
    return null;
  }
}

/**
 * Remove meta_final tag from content string.
 */
export function stripMetaFinal(content: string): string {
  return content.replace(/<meta_final\s+data="[^"]*"[^>]*><\/meta_final>/, '');
}

/**
 * Create a meta_final consumer that buffers partial tags across chunks.
 * Returns a function that strips meta_final tags from incoming content and
 * invokes the provided callback with parsed metadata.
 */
export function createMetaFinalConsumer(
  onMetaFinal: (meta: MetaFinal) => void
): (chunk: string) => string {
  let buffer = '';
  
  return (chunk: string) => {
    if (!chunk) return '';
    buffer += chunk;
    let output = '';
    
    while (buffer.length > 0) {
      const start = buffer.indexOf('<meta_final');
      if (start === -1) {
        output += buffer;
        buffer = '';
        break;
      }
      
      if (start > 0) {
        output += buffer.slice(0, start);
        buffer = buffer.slice(start);
      }
      
      const end = buffer.indexOf('</meta_final>');
      if (end === -1) {
        break;
      }
      
      const endIndex = end + '</meta_final>'.length;
      const tag = buffer.slice(0, endIndex);
      const parsed = parseMetaFinal(tag);
      if (parsed) {
        onMetaFinal(parsed);
      }
      buffer = buffer.slice(endIndex);
    }
    
    return output;
  };
}

