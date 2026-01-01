import type { AgentNode } from '@/lib/parsers';
import { TextBlock } from './text-block';
import { ThinkingBlock } from './thinking-block';
import { ToolCallBlock } from './tool-call-block';
import { ToolResultBlock } from './tool-result-block';
import { FilesBlock } from './files-block';
import { CitationBlock } from './citation-block';

interface NodeRendererProps {
  node: AgentNode;
}

/**
 * Extract citation data from a citations node's children.
 */
function extractCitations(node: AgentNode): Array<{
  type?: string;
  documentIndex?: string;
  startPage?: string;
  endPage?: string;
  url?: string;
  title?: string;
  citedText?: string;
}> {
  if (!node.children) return [];
  
  return node.children
    .filter(child => child.type === 'element' && child.tagName === 'citation')
    .map(citation => ({
      type: citation.attributes?.type,
      documentIndex: citation.attributes?.document_index,
      startPage: citation.attributes?.start_page_number,
      endPage: citation.attributes?.end_page_number,
      url: citation.attributes?.url,
      title: citation.attributes?.document_title || citation.attributes?.title,
      citedText: citation.children?.[0]?.content,
    }));
}

/**
 * Recursively renders an AgentNode tree.
 * Maps tagNames to appropriate block components.
 */
export function NodeRenderer({ node }: NodeRendererProps) {
  // Handle text nodes
  if (node.type === 'text') {
    return <TextBlock content={node.content || ''} />;
  }

  // Handle element nodes
  const children = node.children?.map((child, index) => (
    <NodeRenderer key={index} node={child} />
  ));

  switch (node.tagName) {
    case 'thinking':
      return <ThinkingBlock>{children}</ThinkingBlock>;

    case 'tool_call':
      return (
        <ToolCallBlock
          toolName={node.attributes?.name}
          toolId={node.attributes?.id}
        >
          {children}
        </ToolCallBlock>
      );

    case 'server_tool_call':
      return (
        <ToolCallBlock
          toolName={node.attributes?.name}
          toolId={node.attributes?.id}
          isServer={true}
        >
          {children}
        </ToolCallBlock>
      );

    case 'tool_result':
      return (
        <ToolResultBlock 
          isError={node.attributes?.is_error === 'true'}
          resultName={node.attributes?.name}
        >
          {children}
        </ToolResultBlock>
      );

    case 'server_tool_result':
      return (
        <ToolResultBlock
          isError={node.attributes?.is_error === 'true'}
          resultName={node.attributes?.name}
          toolType={node.attributes?.toolType}
          isServer={true}
        >
          {children}
        </ToolResultBlock>
      );

    case 'error':
      return (
        <ToolResultBlock isError={true}>
          {children}
        </ToolResultBlock>
      );

    case 'text':
      // Text element wrapper - use span for inline flow with citations
      return <span className="inline">{children}</span>;

    case 'citations':
      // Extract citation data and render inline markers
      return <CitationBlock citations={extractCitations(node)} />;

    case 'citation':
      // Individual citations should be handled by parent 'citations' node
      // If encountered directly, skip
      return null;

    case 'meta_init':
      // Skip rendering meta_init
      return null;

    case 'meta_files':
      // Render generated files block
      const filesContent = node.children?.[0]?.content || '';
      return <FilesBlock content={filesContent} />;

    case 'chart':
    case 'table':
      // Placeholder for future chart/table implementations
      return (
        <div className="my-2 p-3 border border-zinc-200 dark:border-zinc-700 rounded-lg bg-zinc-50 dark:bg-zinc-900">
          <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-2">
            {node.tagName === 'chart' ? '📊 Chart' : '📋 Table'}
          </div>
          {children}
        </div>
      );

    case 'root':
      // Root node - just render children
      return <>{children}</>;

    default:
      // Unknown tags - render children with a subtle wrapper
      if (children && children.length > 0) {
        return <div className="my-1">{children}</div>;
      }
      return null;
  }
}

interface NodeListRendererProps {
  nodes: AgentNode[];
}

/**
 * Renders a list of AgentNodes.
 */
export function NodeListRenderer({ nodes }: NodeListRendererProps) {
  return (
    <>
      {nodes.map((node, index) => (
        <NodeRenderer key={index} node={node} />
      ))}
    </>
  );
}
