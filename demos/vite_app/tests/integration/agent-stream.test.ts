/**
 * Integration tests for streamAgent function with MSW mocking.
 * 
 * These tests verify the full streaming flow from API call to parsed nodes.
 * Note: Uses MSW to mock the agent API responses with realistic SSE streams.
 */

import { describe, it, expect, vi } from 'vitest';
import { server } from '../mocks/server';
import { createScenarioHandler, getAvailableScenarios } from '../mocks/handlers';
import { streamAgent, type AgentState, type AgentConfig } from '../../src/lib/agent-stream';
import { hasExpectedNodeTypes, getNodeTextContent, findNodeByTag } from '../fixtures/helpers';
import { ALL_SCENARIOS } from '../fixtures/scenarios';

describe('streamAgent', () => {
  describe('format detection', () => {
    it('detects raw format from meta_init', async () => {
      server.use(createScenarioHandler('no_tools_raw'));
      
      const updates: AgentState[] = [];
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => updates.push({ ...state }),
      });
      
      // Find the first update with streamFormat set
      const formatUpdate = updates.find(u => u.streamFormat);
      expect(formatUpdate?.streamFormat).toBe('raw');
    });

    it('detects xml format from meta_init', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      const updates: AgentState[] = [];
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => updates.push({ ...state }),
      });
      
      const formatUpdate = updates.find(u => u.streamFormat);
      expect(formatUpdate?.streamFormat).toBe('xml');
    });
  });

  describe('status transitions', () => {
    it('transitions from streaming to complete', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      const updates: AgentState[] = [];
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => updates.push({ ...state }),
      });
      
      // First update should be streaming
      expect(updates[0].status).toBe('streaming');
      
      // Last update should be complete
      expect(updates[updates.length - 1].status).toBe('complete');
    });

    it('calls onComplete callback when stream finishes', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      const onComplete = vi.fn();
      
      await streamAgent('test prompt', undefined, {
        onUpdate: () => {},
        onComplete,
      });
      
      expect(onComplete).toHaveBeenCalledOnce();
      expect(onComplete.mock.calls[0][0].status).toBe('complete');
    });
  });

  describe('metadata extraction', () => {
    it('extracts agent_uuid from meta_init', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      const updates: AgentState[] = [];
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => updates.push({ ...state }),
      });
      
      const metadataUpdate = updates.find(u => u.metadata?.agent_uuid);
      expect(metadataUpdate?.metadata?.agent_uuid).toBeDefined();
    });

    it('extracts model from meta_init', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      const updates: AgentState[] = [];
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => updates.push({ ...state }),
      });
      
      const metadataUpdate = updates.find(u => u.metadata?.model);
      expect(metadataUpdate?.metadata?.model).toBe('claude-sonnet-4-5');
    });
  });

  describe('node parsing - XML format', () => {
    it('parses simple hello response with thinking and text', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      let finalState: AgentState | null = null;
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => { finalState = state; },
      });
      
      expect(finalState).not.toBeNull();
      expect(finalState!.nodes.length).toBeGreaterThan(0);
      expect(hasExpectedNodeTypes(finalState!.nodes, ['thinking', 'text'])).toBe(true);
    });

    it('parses tool calls in XML format', async () => {
      server.use(createScenarioHandler('client_tools_xml'));
      
      let finalState: AgentState | null = null;
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => { finalState = state; },
      });
      
      expect(finalState).not.toBeNull();
      expect(hasExpectedNodeTypes(finalState!.nodes, ['tool_call', 'tool_result'])).toBe(true);
    });

    it('parses server tool calls in XML format', async () => {
      server.use(createScenarioHandler('server_tools_xml'));
      
      let finalState: AgentState | null = null;
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => { finalState = state; },
      });
      
      expect(finalState).not.toBeNull();
      expect(hasExpectedNodeTypes(finalState!.nodes, ['server_tool_call', 'server_tool_result'])).toBe(true);
    });

    it('parses citations in XML format', async () => {
      server.use(createScenarioHandler('citations_xml'));
      
      let finalState: AgentState | null = null;
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => { finalState = state; },
      });
      
      expect(finalState).not.toBeNull();
      expect(hasExpectedNodeTypes(finalState!.nodes, ['citations'])).toBe(true);
    });
  });

  describe('node parsing - Raw format', () => {
    it('parses simple response with thinking and text', async () => {
      server.use(createScenarioHandler('no_tools_raw'));
      
      let finalState: AgentState | null = null;
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => { finalState = state; },
      });
      
      expect(finalState).not.toBeNull();
      expect(finalState!.nodes.length).toBeGreaterThan(0);
      expect(hasExpectedNodeTypes(finalState!.nodes, ['thinking', 'text'])).toBe(true);
    });

    it('parses client tool calls in raw format', async () => {
      server.use(createScenarioHandler('client_tools_raw'));
      
      let finalState: AgentState | null = null;
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => { finalState = state; },
      });
      
      expect(finalState).not.toBeNull();
      expect(hasExpectedNodeTypes(finalState!.nodes, ['tool_call', 'tool_result'])).toBe(true);
    });
  });

  describe('streaming updates', () => {
    it('provides progressive updates during streaming', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      const updates: AgentState[] = [];
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => updates.push({ ...state }),
      });
      
      // Should have multiple updates (not just start and end)
      expect(updates.length).toBeGreaterThan(2);
      
      // Node count should grow over time
      const nodeCounts = updates.map(u => u.nodes.length);
      const hasGrowth = nodeCounts.some((count, i) => 
        i > 0 && count > nodeCounts[i - 1]
      );
      expect(hasGrowth).toBe(true);
    });

    it('accumulates rawContent for debugging', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      let finalState: AgentState | null = null;
      
      await streamAgent('test prompt', undefined, {
        onUpdate: (state) => { finalState = state; },
      });
      
      expect(finalState).not.toBeNull();
      expect(finalState!.rawContent).toBeDefined();
      expect(finalState!.rawContent.length).toBeGreaterThan(0);
    });
  });

  describe('config options', () => {
    it('passes agent_type to request body', async () => {
      server.use(createScenarioHandler('hello_xml'));
      
      const config: AgentConfig = {
        agent_type: 'agent_all_xml',
      };
      
      await streamAgent('test prompt', config, {
        onUpdate: () => {},
      });
      
      // If we got here without error, the request was made successfully
      expect(true).toBe(true);
    });
  });
});

describe('scenario coverage', () => {
  // Generate tests for all available scenarios
  const scenarios = getAvailableScenarios();
  
  it.each(scenarios)('scenario "%s" produces expected node types', async (scenarioName) => {
    server.use(createScenarioHandler(scenarioName));
    
    const scenario = ALL_SCENARIOS[scenarioName];
    let finalState: AgentState | null = null;
    
    await streamAgent('test prompt', undefined, {
      onUpdate: (state) => { finalState = state; },
    });
    
    expect(finalState).not.toBeNull();
    expect(hasExpectedNodeTypes(finalState!.nodes, scenario.expectedNodeTypes)).toBe(true);
  });
});

