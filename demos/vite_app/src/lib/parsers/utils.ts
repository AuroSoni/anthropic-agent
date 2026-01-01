/**
 * Shared utilities for parser modules.
 */

/**
 * Decode common HTML entities in a string.
 */
export function decodeHtmlEntities(str: string): string {
  return str
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&amp;/g, '&')
    .replace(/&#x27;/g, "'")
    .replace(/&#39;/g, "'")
    .replace(/&#(\d+);/g, (_, dec) => String.fromCharCode(parseInt(dec, 10)))
    .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
}

/**
 * Unescape SSE-escaped newlines in text content.
 * Converts \n (two chars) to actual newline.
 */
export function unescapeSseNewlines(str: string): string {
  return str.replace(/\\n/g, '\n');
}

