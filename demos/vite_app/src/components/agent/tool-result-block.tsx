import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, CheckCircle2, XCircle, Globe } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface ToolResultBlockProps {
  isError?: boolean;
  resultName?: string;
  toolType?: string;
  isServer?: boolean;
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
 * Server tools (isServer=true) use blue styling when not error.
 */
export function ToolResultBlock({ isError = false, resultName, toolType, isServer = false, children }: ToolResultBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Color scheme: red for errors, blue for server success, emerald for client success
  const getColors = () => {
    if (isError) {
      return {
        border: 'border-red-200 dark:border-red-800',
        bg: 'bg-red-50/50 dark:bg-red-950/30',
        text: 'text-red-700 dark:text-red-300',
        hover: 'hover:bg-red-100/50 dark:hover:bg-red-900/30',
        badge: 'bg-red-200 text-red-800 dark:bg-red-800 dark:text-red-200',
        content: 'text-red-900 dark:text-red-100',
      };
    }
    if (isServer) {
      return {
        border: 'border-blue-200 dark:border-blue-800',
        bg: 'bg-blue-50/50 dark:bg-blue-950/30',
        text: 'text-blue-700 dark:text-blue-300',
        hover: 'hover:bg-blue-100/50 dark:hover:bg-blue-900/30',
        badge: 'bg-blue-200 text-blue-800 dark:bg-blue-800 dark:text-blue-200',
        content: 'text-blue-900 dark:text-blue-100',
      };
    }
    return {
      border: 'border-emerald-200 dark:border-emerald-800',
      bg: 'bg-emerald-50/50 dark:bg-emerald-950/30',
      text: 'text-emerald-700 dark:text-emerald-300',
      hover: 'hover:bg-emerald-100/50 dark:hover:bg-emerald-900/30',
      badge: 'bg-emerald-200 text-emerald-800 dark:bg-emerald-800 dark:text-emerald-200',
      content: 'text-emerald-900 dark:text-emerald-100',
    };
  };

  const colors = getColors();
  
  // Use toolType if available, otherwise resultName
  const displayName = toolType 
    ? formatResultName(toolType) 
    : (resultName ? formatResultName(resultName) : 'Tool Result');
  
  const label = isServer ? 'Server Result' : 'Result';
  const Icon = isError ? XCircle : (isServer ? Globe : CheckCircle2);

  return (
    <div className={`my-2 rounded-lg border ${colors.border} ${colors.bg}`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className={`flex w-full items-center gap-2 px-3 py-2 text-left text-sm font-medium ${colors.text} ${colors.hover}`}
      >
        {isExpanded ? (
          <ChevronDown className="h-4 w-4" />
        ) : (
          <ChevronRight className="h-4 w-4" />
        )}
        <Icon className="h-4 w-4" />
        <span>{label}</span>
        {(resultName || toolType) && (
          <Badge variant="secondary" className={`ml-1 ${colors.badge}`}>
            {displayName}
          </Badge>
        )}
        {isError && <span className="text-xs">(Error)</span>}
      </button>
      {isExpanded && (
        <div className={`border-t ${colors.border} px-3 py-2 text-sm`}>
          <pre className={`overflow-x-auto whitespace-pre-wrap break-words font-mono text-xs ${colors.content}`}>
            {children}
          </pre>
        </div>
      )}
    </div>
  );
}

