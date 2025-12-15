import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface TextBlockProps {
  content: string;
}

/**
 * TextBlock renders GitHub Flavored Markdown content.
 * Supports tables, strikethrough, task lists, autolinks, and standard markdown.
 */
export function TextBlock({ content }: TextBlockProps) {
  return (
    <div className="prose prose-zinc dark:prose-invert max-w-none">
      <Markdown remarkPlugins={[remarkGfm]}>
        {content}
      </Markdown>
    </div>
  );
}

