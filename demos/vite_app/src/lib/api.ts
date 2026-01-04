/**
 * API client for conversation history and other backend endpoints.
 */

const API_BASE = 'http://localhost:8000';

/**
 * A single conversation turn from the history API.
 */
export interface ConversationItem {
  conversation_id: string;
  run_id: string;
  sequence_number: number;
  user_message: string;
  final_response: string | null;
  started_at: string | null;
  completed_at: string | null;
  stop_reason: string | null;
  total_steps: number | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
  } | null;
  generated_files: Array<{
    file_id: string;
    filename: string;
    mime_type: string;
  }> | null;
  messages: unknown[];
}

/**
 * Response from the conversation history endpoint.
 */
export interface ConversationHistoryResponse {
  conversations: ConversationItem[];
  has_more: boolean;
}

/**
 * Fetch conversation history for an agent with cursor-based pagination.
 * 
 * @param agentUuid - The agent's UUID
 * @param before - Load conversations with sequence_number < before (undefined = latest)
 * @param limit - Maximum number of conversations to return (default 20, max 100)
 * @param agentType - Agent type to determine database backend (default 'agent_frontend_tools')
 * @returns Conversation history with has_more flag
 */
export async function fetchConversationHistory(
  agentUuid: string,
  before?: number,
  limit: number = 20,
  agentType: string = 'agent_frontend_tools'
): Promise<ConversationHistoryResponse> {
  const params = new URLSearchParams();
  if (before !== undefined) {
    params.set('before', String(before));
  }
  params.set('limit', String(limit));
  params.set('agent_type', agentType);

  const url = `${API_BASE}/agent/${agentUuid}/conversations?${params.toString()}`;
  
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch conversation history: ${response.status} ${response.statusText}`);
  }
  
  return response.json();
}

