import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, CheckCircle2, XCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface ToolResultBlockProps {
  isError?: boolean;
  resultName?: string;
  children: ReactNode;
}

/**
 * Format a result name for display (e.g., "bash_code_execution_tool_result" -> "Bash Code Execution")
 */
function formatResultName(name: string): string {
  return name
    .replace(/_tool_result$/, '')
    .replace(/content-block-/, '')
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * ToolResultBlock renders tool execution results.
 * Collapsed by default.
 */
export function ToolResultBlock({ isError = false, resultName, children }: ToolResultBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const borderColor = isError
    ? 'border-red-200 dark:border-red-800'
    : 'border-emerald-200 dark:border-emerald-800';
  const bgColor = isError
    ? 'bg-red-50/50 dark:bg-red-950/30'
    : 'bg-emerald-50/50 dark:bg-emerald-950/30';
  const textColor = isError
    ? 'text-red-700 dark:text-red-300'
    : 'text-emerald-700 dark:text-emerald-300';
  const hoverBg = isError
    ? 'hover:bg-red-100/50 dark:hover:bg-red-900/30'
    : 'hover:bg-emerald-100/50 dark:hover:bg-emerald-900/30';
  const contentTextColor = isError
    ? 'text-red-900 dark:text-red-100'
    : 'text-emerald-900 dark:text-emerald-100';

  const displayName = resultName ? formatResultName(resultName) : 'Tool Result';

  return (
    <div className={`my-2 rounded-lg border ${borderColor} ${bgColor}`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium ${textColor} ${hoverBg}`}
      >
        {isExpanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        {isError ? (
          <XCircle className="h-4 w-4" />
        ) : (
          <CheckCircle2 className="h-4 w-4" />
        )}
        <span>Result</span>
        {resultName && (
          <Badge 
            variant="secondary" 
            className={isError 
              ? "ml-1 bg-red-200 text-red-800 dark:bg-red-800 dark:text-red-200"
              : "ml-1 bg-emerald-200 text-emerald-800 dark:bg-emerald-800 dark:text-emerald-200"
            }
          >
            {displayName}
          </Badge>
        )}
        {isError && <span className="text-xs">(Error)</span>}
      </button>
      {isExpanded && (
        <div className={`border-t ${borderColor} px-3 py-2 text-sm`}>
          <pre className={`overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs ${contentTextColor}`}>
            {children}
          </pre>
        </div>
      )}
    </div>
  );
}

