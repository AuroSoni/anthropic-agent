/**
 * StreamingIndicator shows an animated indicator while streaming is in progress.
 * Uses a pulsing dot animation.
 */
export function StreamingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-2 text-sm text-zinc-500 dark:text-zinc-400">
      <span className="flex gap-1">
        <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" style={{ animationDelay: '0ms' }} />
        <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" style={{ animationDelay: '150ms' }} />
        <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" style={{ animationDelay: '300ms' }} />
      </span>
      <span className="ml-1">Streaming...</span>
    </div>
  );
}

