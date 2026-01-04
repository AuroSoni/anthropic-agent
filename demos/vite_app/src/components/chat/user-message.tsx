import { cn } from '@/lib/utils';

interface UserMessageProps {
  message: string;
  timestamp?: string;
  className?: string;
}

/**
 * User message bubble - right-aligned with distinct styling.
 */
export function UserMessage({ message, timestamp, className }: UserMessageProps) {
  return (
    <div className={cn('flex justify-end', className)}>
      <div className="max-w-[80%] space-y-1">
        <div className="rounded-2xl rounded-br-md bg-blue-600 px-4 py-2.5 text-white shadow-sm">
          <p className="whitespace-pre-wrap text-sm">{message}</p>
        </div>
        {timestamp && (
          <p className="text-right text-xs text-zinc-400 dark:text-zinc-500">
            {formatTimestamp(timestamp)}
          </p>
        )}
      </div>
    </div>
  );
}

function formatTimestamp(isoString: string): string {
  try {
    const date = new Date(isoString);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

