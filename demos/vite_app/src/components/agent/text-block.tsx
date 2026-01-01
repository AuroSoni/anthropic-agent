import { useState, type ComponentPropsWithoutRef } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { FileText } from 'lucide-react';

interface TextBlockProps {
  content: string;
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

interface CiteProps extends ComponentPropsWithoutRef<'cite'> {
  'data-index'?: string;
  doc?: string;
  startPage?: string;
  endPage?: string;
  url?: string;
  title?: string;
  type?: string;
  children?: React.ReactNode;
}

/**
 * Custom cite component that renders inline citation markers with hover tooltips.
 * Index is read from the data-index attribute set during the merge phase.
 */
function Cite({ 'data-index': dataIndex, doc, startPage, endPage, url, title, type, children, ...props }: CiteProps) {
  const [isHovered, setIsHovered] = useState(false);
  
  // Read index from data-index attribute (assigned during merge phase)
  const index = dataIndex ? parseInt(dataIndex, 10) : 1;
  
  const pageRange = formatPageRange(startPage, endPage);
  const citedText = typeof children === 'string' ? children : '';
  const hasDetails = pageRange || title || citedText;

  return (
    <span 
      className="relative inline-block"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      {...props}
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
            {title ? (
              <span className="truncate">{title}</span>
            ) : (
              <span>Document {doc || index}</span>
            )}
            {pageRange && (
              <span className="ml-auto text-zinc-500 dark:text-zinc-400 whitespace-nowrap">
                {pageRange}
              </span>
            )}
          </div>

          {/* Cited text preview */}
          {citedText && (
            <p className="text-zinc-500 dark:text-zinc-400 italic border-l-2 border-zinc-300 dark:border-zinc-600 pl-2 mt-1">
              "{truncateText(citedText, 150)}"
            </p>
          )}

          {/* URL link if available */}
          {url && (
            <a 
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 dark:text-blue-400 hover:underline mt-1 block truncate"
            >
              {url}
            </a>
          )}
        </div>
      )}
    </span>
  );
}

/**
 * Custom components for react-markdown to handle cite tags and other elements.
 */
const markdownComponents: Components = {
  cite: Cite as Components['cite'],
};

/**
 * TextBlock renders GitHub Flavored Markdown content with inline citation support.
 * Supports tables, strikethrough, task lists, autolinks, standard markdown,
 * and custom <cite> tags for inline citations.
 * 
 * Citation indices are pre-assigned during the merge phase in NodeListRenderer
 * and passed via the data-index attribute on each <cite> tag.
 */
export function TextBlock({ content }: TextBlockProps) {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <Markdown 
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={markdownComponents}
      >
        {content}
      </Markdown>
    </div>
  );
}
