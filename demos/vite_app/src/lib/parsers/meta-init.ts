/**
 * Utilities for the backend `<meta_init ...>` header: detect/parse/strip initialization metadata
 * (e.g., chosen stream format) before routing the stream to the correct parser.
 */
import type { MetaInit } from './types';

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
    // Unescape HTML entities and parse JSON
    const escapedJson = match[1];
    const json = escapedJson
      .replace(/&quot;/g, '"')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&#x27;/g, "'")
      .replace(/&#39;/g, "'");
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

