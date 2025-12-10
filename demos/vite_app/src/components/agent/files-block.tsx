import { FileDown, FileIcon, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface GeneratedFile {
  file_id: string;
  filename: string;
  storage_location: string;
  size: number;
  created_at?: string;
  is_update?: boolean;
}

interface FilesBlockProps {
  content: string;
}

/**
 * Format file size in human-readable format
 */
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Get file extension for icon styling
 */
function getFileExtension(filename: string): string {
  const ext = filename.split('.').pop()?.toLowerCase() || '';
  return ext;
}

/**
 * Get badge color based on file type
 */
function getExtensionColor(ext: string): string {
  const colors: Record<string, string> = {
    pdf: 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300',
    docx: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
    doc: 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
    xlsx: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    xls: 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
    png: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
    jpg: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
    jpeg: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
    gif: 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
    svg: 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300',
    zip: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300',
    csv: 'bg-teal-100 text-teal-700 dark:bg-teal-900 dark:text-teal-300',
    json: 'bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300',
    txt: 'bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300',
  };
  return colors[ext] || 'bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300';
}

/**
 * FilesBlock displays generated files with download buttons.
 */
export function FilesBlock({ content }: FilesBlockProps) {
  let files: GeneratedFile[] = [];
  
  try {
    const parsed = JSON.parse(content);
    files = parsed.files || [];
  } catch {
    // Invalid JSON - don't render anything
    return null;
  }

  if (files.length === 0) {
    return null;
  }

  return (
    <div className="my-3 rounded-lg border border-sky-200 bg-sky-50/50 dark:border-sky-800 dark:bg-sky-950/30">
      <div className="flex items-center gap-2 border-b border-sky-200 px-3 py-2 dark:border-sky-800">
        <FileIcon className="h-4 w-4 text-sky-600 dark:text-sky-400" />
        <span className="text-sm font-medium text-sky-700 dark:text-sky-300">
          Generated Files
        </span>
        <Badge variant="secondary" className="ml-1 bg-sky-200 text-sky-800 dark:bg-sky-800 dark:text-sky-200">
          {files.length} {files.length === 1 ? 'file' : 'files'}
        </Badge>
      </div>
      
      <div className="divide-y divide-sky-200 dark:divide-sky-800">
        {files.map((file) => {
          const ext = getFileExtension(file.filename);
          
          return (
            <div 
              key={file.file_id}
              className="flex items-center justify-between gap-3 px-3 py-2"
            >
              <div className="flex items-center gap-3 min-w-0 flex-1">
                <FileIcon className="h-5 w-5 shrink-0 text-sky-500 dark:text-sky-400" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-medium text-zinc-900 dark:text-zinc-100">
                      {file.filename}
                    </span>
                    <Badge variant="secondary" className={`shrink-0 text-xs ${getExtensionColor(ext)}`}>
                      {ext.toUpperCase()}
                    </Badge>
                    {file.is_update && (
                      <Badge variant="outline" className="shrink-0 text-xs">
                        Updated
                      </Badge>
                    )}
                  </div>
                  <div className="text-xs text-zinc-500 dark:text-zinc-400">
                    {formatFileSize(file.size)}
                    {file.created_at && (
                      <span className="ml-2">
                        • {new Date(file.created_at).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
              </div>
              
              <div className="flex items-center gap-1 shrink-0">
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 gap-1 text-xs"
                  asChild
                >
                  <a 
                    href={file.storage_location} 
                    download={file.filename}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <FileDown className="h-3.5 w-3.5" />
                    Download
                  </a>
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0"
                  asChild
                >
                  <a 
                    href={file.storage_location}
                    target="_blank"
                    rel="noopener noreferrer"
                    title="Open in new tab"
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                  </a>
                </Button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
