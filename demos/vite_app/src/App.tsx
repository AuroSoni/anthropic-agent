import { useState, useCallback } from 'react';
import { AgentViewer } from './components/agent';
import { ConversationSidebar } from './components/sidebar';
import { useAgentUrlState } from './hooks/use-url-state';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const { agentUuid, setAgentUuid } = useAgentUrlState();
  
  // Key to force AgentViewer remount when conversation changes
  const [viewerKey, setViewerKey] = useState(0);
  
  // Trigger to refresh sidebar sessions list
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleSelectConversation = useCallback((uuid: string) => {
    setAgentUuid(uuid);
    setViewerKey(k => k + 1);
  }, [setAgentUuid]);

  const handleNewConversation = useCallback(() => {
    setAgentUuid(null);
    setViewerKey(k => k + 1);
  }, [setAgentUuid]);

  const handleStreamComplete = useCallback(() => {
    setRefreshTrigger(prev => prev + 1);
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      <ConversationSidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        currentAgentUuid={agentUuid ?? undefined}
        refreshTrigger={refreshTrigger}
      />
      <main className="flex-1 min-w-0 min-h-0 overflow-hidden">
        <AgentViewer key={viewerKey} onStreamComplete={handleStreamComplete} />
      </main>
    </div>
  );
}

export default App;
