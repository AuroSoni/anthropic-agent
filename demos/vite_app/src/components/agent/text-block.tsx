interface TextBlockProps {
  content: string;
}

/**
 * TextBlock renders plain text content.
 * Always visible, no collapse functionality.
 */
export function TextBlock({ content }: TextBlockProps) {
  return (
    <span className="whitespace-pre-wrap break-words">{content}</span>
  );
}

