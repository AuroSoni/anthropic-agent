import { useState } from 'react';
import { MessageCircleQuestion, Check, X, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import type { PendingFrontendTool } from '@/lib/parsers/types';

interface FrontendToolPromptProps {
  tool: PendingFrontendTool;
  onSubmit: (response: string) => void;
  disabled?: boolean;
}

/**
 * Inline card component for frontend tool interactions.
 * Displays the tool request and provides Yes/No buttons for user_confirm.
 * Uses a violet/purple color scheme to distinguish from backend (amber) and server (blue) tools.
 */
export function FrontendToolPrompt({ tool, onSubmit, disabled = false }: FrontendToolPromptProps) {
  const [submitted, setSubmitted] = useState(false);
  const [selectedResponse, setSelectedResponse] = useState<string | null>(null);

  const handleResponse = (response: string) => {
    if (submitted || disabled) return;
    setSubmitted(true);
    setSelectedResponse(response);
    onSubmit(response);
  };

  // Extract message from tool input (for user_confirm tool)
  const message = typeof tool.input?.message === 'string' 
    ? tool.input.message 
    : JSON.stringify(tool.input);

  // Violet/purple color scheme for frontend tools
  const colors = {
    border: 'border-violet-200 dark:border-violet-800',
    bg: 'bg-violet-50/50 dark:bg-violet-950/30',
    text: 'text-violet-700 dark:text-violet-300',
    headerBg: 'bg-violet-100/50 dark:bg-violet-900/30',
    buttonYes: 'bg-emerald-500 hover:bg-emerald-600 text-white',
    buttonNo: 'bg-rose-500 hover:bg-rose-600 text-white',
    buttonDisabled: 'opacity-50 cursor-not-allowed',
  };

  return (
    <div className={`my-3 rounded-lg border-2 ${colors.border} ${colors.bg} overflow-hidden`}>
      {/* Header */}
      <div className={`flex items-center gap-2 px-4 py-2 ${colors.headerBg}`}>
        <MessageCircleQuestion className={`h-5 w-5 ${colors.text}`} />
        <span className={`text-sm font-medium ${colors.text}`}>
          {tool.name === 'user_confirm' ? 'Confirmation Required' : tool.name}
        </span>
        {submitted && (
          <span className="ml-auto flex items-center gap-1 text-xs text-zinc-500 dark:text-zinc-400">
            <Loader2 className="h-3 w-3 animate-spin" />
            Continuing...
          </span>
        )}
      </div>
      
      {/* Message */}
      <div className="px-4 py-3">
        <p className="text-sm text-zinc-700 dark:text-zinc-300 whitespace-pre-wrap">
          {message}
        </p>
      </div>
      
      {/* Actions */}
      <div className="flex gap-2 px-4 pb-3">
        {tool.name === 'user_confirm' ? (
          <>
            <Button
              size="sm"
              onClick={() => handleResponse('yes')}
              disabled={submitted || disabled}
              className={`${colors.buttonYes} ${(submitted || disabled) ? colors.buttonDisabled : ''}`}
            >
              {submitted && selectedResponse === 'yes' ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Check className="h-4 w-4 mr-1" />
              )}
              Yes
            </Button>
            <Button
              size="sm"
              onClick={() => handleResponse('no')}
              disabled={submitted || disabled}
              className={`${colors.buttonNo} ${(submitted || disabled) ? colors.buttonDisabled : ''}`}
            >
              {submitted && selectedResponse === 'no' ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <X className="h-4 w-4 mr-1" />
              )}
              No
            </Button>
          </>
        ) : (
          // Generic text input for other tools (future extension)
          <Button
            size="sm"
            variant="outline"
            onClick={() => handleResponse('ok')}
            disabled={submitted || disabled}
            className={(submitted || disabled) ? colors.buttonDisabled : ''}
          >
            Continue
          </Button>
        )}
      </div>
      
      {/* Tool ID (debug info) */}
      <div className="px-4 pb-2 text-xs text-zinc-400 dark:text-zinc-600">
        Tool ID: {tool.tool_use_id.slice(0, 12)}...
      </div>
    </div>
  );
}

