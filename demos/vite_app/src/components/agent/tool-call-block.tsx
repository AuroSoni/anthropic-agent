import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, Wrench } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface ToolCallBlockProps {
  toolName?: string;
  toolId?: string;
  children: ReactNode;
}

/**
 * ToolCallBlock renders tool call invocations.
 * Collapsed by default, shows tool name in header.
 */
export function ToolCallBlock({ toolName, toolId, children }: ToolCallBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="my-2 rounded-lg border border-amber-200 bg-amber-50/50 dark:border-amber-800 dark:bg-amber-950/30">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium text-amber-700 hover:bg-amber-100/50 dark:text-amber-300 dark:hover:bg-amber-900/30"
      >
        {isExpanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        <Wrench className="h-4 w-4" />
        <span>Tool Call</span>
        {toolName && (
          <Badge variant="secondary" className="ml-1 bg-amber-200 text-amber-800 dark:bg-amber-800 dark:text-amber-200">
            {toolName}
          </Badge>
        )}
        {toolId && (
          <span className="ml-auto text-xs text-amber-500 dark:text-amber-400">
            {toolId.slice(0, 8)}...
          </span>
        )}
      </button>
      {isExpanded && (
        <div className="border-t border-amber-200 px-3 py-2 text-sm dark:border-amber-800">
          <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs text-amber-900 dark:text-amber-100">
            {children}
          </pre>
        </div>
      )}
    </div>
  );
}

