import type { AgentNode } from '@/lib/parsers';
import { TextBlock } from './text-block';
import { ThinkingBlock } from './thinking-block';
import { ToolCallBlock } from './tool-call-block';
import { ToolResultBlock } from './tool-result-block';

interface NodeRendererProps {
  node: AgentNode;
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

    case 'tool_result':
      return (
        <ToolResultBlock 
          isError={node.attributes?.is_error === 'true'}
          resultName={node.attributes?.name}
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
      // Text element wrapper (from raw format)
      return <span className="whitespace-pre-wrap">{children}</span>;

    case 'meta_init':
    case 'meta_files':
      // Skip rendering meta tags
      return null;

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

