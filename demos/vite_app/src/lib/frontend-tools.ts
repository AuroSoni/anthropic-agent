/**
 * Frontend tool registry for browser-executed tools.
 * These tools are schema-only on the server; execution happens in the browser.
 */
import type { PendingFrontendTool, FrontendToolResult } from './parsers/types';

/**
 * Handler function for a frontend tool.
 * Takes the tool definition and user response, returns the result to send back.
 */
export type FrontendToolHandler = (
  tool: PendingFrontendTool,
  userResponse: string
) => FrontendToolResult;

/**
 * Registry of frontend tool handlers.
 * Add new tools here as they are implemented.
 */
export const frontendToolHandlers: Record<string, FrontendToolHandler> = {
  /**
   * user_confirm: Asks the user for yes/no confirmation.
   * The user response ("yes" or "no") is passed directly as the tool result.
   */
  user_confirm: (tool, userResponse) => ({
    tool_use_id: tool.tool_use_id,
    content: userResponse,
  }),
};

/**
 * Create a tool result from a pending tool and user response.
 * Uses the registered handler for the tool, or returns an error if unknown.
 */
export function createToolResult(
  tool: PendingFrontendTool,
  response: string
): FrontendToolResult {
  const handler = frontendToolHandlers[tool.name];
  if (handler) {
    return handler(tool, response);
  }
  return {
    tool_use_id: tool.tool_use_id,
    content: `Unknown frontend tool: ${tool.name}`,
    is_error: true,
  };
}

/**
 * Check if a tool name has a registered frontend handler.
 */
export function hasFrontendHandler(toolName: string): boolean {
  return toolName in frontendToolHandlers;
}

