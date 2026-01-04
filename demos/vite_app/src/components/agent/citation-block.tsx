import { useState } from 'react';
import { FileText } from 'lucide-react';

interface Citation {
  type?: string;
  documentIndex?: string;
  startPage?: string;
  endPage?: string;
  url?: string;
  title?: string;
  citedText?: string;
}

interface CitationBlockProps {
  citations: Citation[];
}

/**
 * Format page range for display.
 */
function formatPageRange(startPage?: string, endPage?: string): string {
  if (!startPage) return '';
  if (!endPage || startPage === endPage) return `p. ${startPage}`;
  return `pp. ${startPage}-${endPage}`;
}

/**
 * Truncate text to a maximum length.
 */
function truncateText(text: string, maxLength: number = 100): string {
  if (text.length <= maxLength) return text;
  return text.slice(0, maxLength).trim() + '...';
}

/**
 * CitationBlock renders inline citation markers with hover tooltips.
 * Each citation appears as a superscript link-style marker.
 */
export function CitationBlock({ citations }: CitationBlockProps) {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <span className="inline-flex items-baseline gap-0.5">
      {citations.map((citation, index) => (
        <CitationMarker key={index} citation={citation} index={index + 1} />
      ))}
    </span>
  );
}

interface CitationMarkerProps {
  citation: Citation;
  index: number;
}

/**
 * Individual citation marker with hover tooltip.
 */
function CitationMarker({ citation, index }: CitationMarkerProps) {
  const [isHovered, setIsHovered] = useState(false);

  const pageRange = formatPageRange(citation.startPage, citation.endPage);
  const hasDetails = pageRange || citation.title || citation.citedText;

  return (
    <span 
      className="relative inline-block"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Citation marker */}
      <span className="cursor-help text-xs text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 font-medium align-super">
        [{index}]
      </span>

      {/* Hover tooltip */}
      {isHovered && hasDetails && (
        <div className="absolute left-0 bottom-full mb-1 z-50 w-64 max-w-xs p-2 text-xs bg-white dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-md shadow-lg">
          {/* Header with page range */}
          <div className="flex items-center gap-1.5 text-zinc-600 dark:text-zinc-300 font-medium mb-1">
            <FileText className="h-3 w-3" />
            {citation.title ? (
              <span className="truncate">{citation.title}</span>
            ) : (
              <span>Document {citation.documentIndex || index}</span>
            )}
            {pageRange && (
              <span className="ml-auto text-zinc-500 dark:text-zinc-400 whitespace-nowrap">
                {pageRange}
              </span>
            )}
          </div>

          {/* Cited text preview */}
          {citation.citedText && (
            <p className="text-zinc-500 dark:text-zinc-400 italic border-l-2 border-zinc-300 dark:border-zinc-600 pl-2 mt-1">
              "{truncateText(citation.citedText, 150)}"
            </p>
          )}

          {/* URL link if available */}
          {citation.url && (
            <a 
              href={citation.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 dark:text-blue-400 hover:underline mt-1 block truncate"
            >
              {citation.url}
            </a>
          )}
        </div>
      )}
    </span>
  );
}

