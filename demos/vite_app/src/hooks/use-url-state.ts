import { useState, useEffect, useCallback } from 'react';

/**
 * Hook for managing agent_uuid in URL query parameters.
 * Enables shareable conversation links via ?agent=uuid
 */
export function useAgentUrlState() {
  const [agentUuid, setAgentUuidState] = useState<string | null>(() => {
    // Initialize from URL on mount
    const params = new URLSearchParams(window.location.search);
    return params.get('agent');
  });

  // Update URL when agentUuid changes
  const setAgentUuid = useCallback((uuid: string | null) => {
    setAgentUuidState(uuid);
    
    const url = new URL(window.location.href);
    if (uuid) {
      url.searchParams.set('agent', uuid);
    } else {
      url.searchParams.delete('agent');
    }
    
    // Update URL without page reload
    window.history.replaceState({}, '', url.toString());
  }, []);

  // Listen for browser back/forward navigation
  useEffect(() => {
    const handlePopState = () => {
      const params = new URLSearchParams(window.location.search);
      setAgentUuidState(params.get('agent'));
    };

    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  return {
    agentUuid,
    setAgentUuid,
    hasAgent: agentUuid !== null,
  };
}

