import { useMemo } from 'react';
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
 * Check if a node is an inline node that should be merged with adjacent inline nodes.
 * Inline nodes: text nodes, text elements, and citations.
 */
function isInlineNode(node: AgentNode): boolean {
  if (node.type === 'text') return true;
  if (node.type === 'element') {
    return node.tagName === 'text' || node.tagName === 'citations';
  }
  return false;
}

/**
 * Escape HTML special characters in text content.
 */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Counter object passed through merge functions to track citation indices */
interface CitationCounter {
  value: number;
}

/**
 * Convert a single citation element to an HTML <cite> tag with index.
 */
function citationToHtml(citation: AgentNode, counter: CitationCounter): string {
  counter.value += 1;
  const attrs: string[] = [`data-index="${counter.value}"`];
  
  if (citation.attributes?.type) {
    attrs.push(`type="${escapeHtml(citation.attributes.type)}"`);
  }
  if (citation.attributes?.document_index) {
    attrs.push(`doc="${escapeHtml(citation.attributes.document_index)}"`);
  }
  if (citation.attributes?.start_page_number) {
    attrs.push(`startPage="${escapeHtml(citation.attributes.start_page_number)}"`);
  }
  if (citation.attributes?.end_page_number) {
    attrs.push(`endPage="${escapeHtml(citation.attributes.end_page_number)}"`);
  }
  if (citation.attributes?.url) {
    attrs.push(`url="${escapeHtml(citation.attributes.url)}"`);
  }
  const title = citation.attributes?.document_title || citation.attributes?.title;
  if (title) {
    attrs.push(`title="${escapeHtml(title)}"`);
  }
  
  const citedText = citation.children?.[0]?.content || '';
  const attrStr = ' ' + attrs.join(' ');
  
  return `<cite${attrStr}>${escapeHtml(citedText)}</cite>`;
}

/**
 * Convert a citations element (containing multiple citation children) to HTML.
 */
function citationsToHtml(node: AgentNode, counter: CitationCounter): string {
  if (!node.children) return '';
  
  return node.children
    .filter(child => child.type === 'element' && child.tagName === 'citation')
    .map(child => citationToHtml(child, counter))
    .join('');
}

/**
 * Recursively extract text content from a node tree.
 */
function nodeToMarkdown(node: AgentNode, counter: CitationCounter): string {
  if (node.type === 'text') {
    return node.content || '';
  }
  
  if (node.type === 'element') {
    if (node.tagName === 'citations') {
      return citationsToHtml(node, counter);
    }
    if (node.tagName === 'text' && node.children) {
      return node.children.map(child => nodeToMarkdown(child, counter)).join('');
    }
  }
  
  return '';
}

/**
 * Merge consecutive inline nodes starting from startIndex.
 * Returns the merged markdown content and the number of nodes consumed.
 */
function mergeInlineNodes(nodes: AgentNode[], startIndex: number, counter: CitationCounter): { content: string; consumed: number } {
  let content = '';
  let consumed = 0;
  
  for (let i = startIndex; i < nodes.length; i++) {
    const node = nodes[i];
    if (!isInlineNode(node)) break;
    
    content += nodeToMarkdown(node, counter);
    consumed++;
  }
  
  return { content, consumed };
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
 * Renders a list of AgentNodes, merging adjacent inline nodes (text + citations)
 * into single TextBlock components for proper inline flow.
 * 
 * Citation indices are assigned during the merge phase and embedded as data-index
 * attributes in the HTML, ensuring stable numbering across re-renders.
 */
export function NodeListRenderer({ nodes }: NodeListRendererProps) {
  // Memoize the element building to ensure stable citation indices
  // Only rebuilds when nodes array changes
  const elements = useMemo(() => {
    const counter: CitationCounter = { value: 0 };
    const result: React.ReactNode[] = [];
    let i = 0;
    
    while (i < nodes.length) {
      const node = nodes[i];
      
      // Check if this starts a sequence of inline nodes
      if (isInlineNode(node)) {
        const { content, consumed } = mergeInlineNodes(nodes, i, counter);
        if (content) {
          result.push(<TextBlock key={i} content={content} />);
        }
        i += consumed;
      } else {
        // Non-inline node - render normally
        result.push(<NodeRenderer key={i} node={node} />);
        i++;
      }
    }
    
    return result;
  }, [nodes]);
  
  return <>{elements}</>;
}
