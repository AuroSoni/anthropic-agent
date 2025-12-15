import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, Brain } from 'lucide-react';

interface ThinkingBlockProps {
  children: ReactNode;
}

/**
 * ThinkingBlock renders thinking/reasoning content.
 * Collapsible but expanded by default.
 */
export function ThinkingBlock({ children }: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="my-2 rounded-lg border border-violet-200 bg-violet-50/50 dark:border-violet-800 dark:bg-violet-950/30">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium text-violet-700 hover:bg-violet-100/50 dark:text-violet-300 dark:hover:bg-violet-900/30"
      >
        {isExpanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        <Brain className="h-4 w-4" />
        <span>Thinking</span>
      </button>
      {isExpanded && (
        <div className="border-t border-violet-200 px-3 py-2 text-sm text-violet-900 dark:border-violet-800 dark:text-violet-100">
          {children}
        </div>
      )}
    </div>
  );
}

