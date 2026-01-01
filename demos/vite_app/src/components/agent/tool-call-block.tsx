import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight, Wrench, Globe } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

interface ToolCallBlockProps {
  toolName?: string;
  toolId?: string;
  isServer?: boolean;
  children: ReactNode;
}

/**
 * ToolCallBlock renders tool call invocations.
 * Collapsed by default, shows tool name in header.
 * Server tools (isServer=true) use blue styling, client tools use amber.
 */
export function ToolCallBlock({ toolName, toolId, isServer = false, children }: ToolCallBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  // Color scheme: blue for server tools, amber for client tools
  const colors = isServer
    ? {
        border: 'border-blue-200 dark:border-blue-800',
        bg: 'bg-blue-50/50 dark:bg-blue-950/30',
        text: 'text-blue-700 dark:text-blue-300',
        hover: 'hover:bg-blue-100/50 dark:hover:bg-blue-900/30',
        badge: 'bg-blue-200 text-blue-800 dark:bg-blue-800 dark:text-blue-200',
        id: 'text-blue-500 dark:text-blue-400',
        content: 'text-blue-900 dark:text-blue-100',
      }
    : {
        border: 'border-amber-200 dark:border-amber-800',
        bg: 'bg-amber-50/50 dark:bg-amber-950/30',
        text: 'text-amber-700 dark:text-amber-300',
        hover: 'hover:bg-amber-100/50 dark:hover:bg-amber-900/30',
        badge: 'bg-amber-200 text-amber-800 dark:bg-amber-800 dark:text-amber-200',
        id: 'text-amber-500 dark:text-amber-400',
        content: 'text-amber-900 dark:text-amber-100',
      };

  const Icon = isServer ? Globe : Wrench;
  const label = isServer ? 'Server Tool' : 'Tool Call';

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
        {toolName && (
          <Badge variant="secondary" className={`ml-1 ${colors.badge}`}>
            {toolName}
          </Badge>
        )}
        {toolId && (
          <span className={`ml-auto text-xs ${colors.id}`}>
            {toolId.slice(0, 8)}...
          </span>
        )}
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

