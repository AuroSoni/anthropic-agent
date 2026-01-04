/**
 * MSW request handlers for mocking the agent API.
 */

import { http, HttpResponse, delay } from 'msw';
import { getScenario, ALL_SCENARIOS } from '../fixtures';

/**
 * Default scenario to use if none specified
 */
const DEFAULT_SCENARIO = 'hello_xml';

/**
 * MSW handlers for agent API
 */
export const handlers = [
  /**
   * POST /agent/run - Main agent streaming endpoint
   * 
   * Use `_test_scenario` in the request body to select a scenario:
   * { user_prompt: "...", agent_type: "...", _test_scenario: "hello_xml" }
   */
  http.post('*/agent/run', async ({ request }) => {
    // Parse request body to get scenario name
    let scenarioName = DEFAULT_SCENARIO;
    
    try {
      const body = await request.json() as Record<string, unknown>;
      if (body._test_scenario && typeof body._test_scenario === 'string') {
        scenarioName = body._test_scenario;
      }
    } catch {
      // Use default scenario if body parsing fails
    }

    const scenario = getScenario(scenarioName);
    
    // Small delay to simulate network
    await delay(10);
    
    // Return SSE stream
    return new HttpResponse(scenario.stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });
  }),

  /**
   * GET /agent/health - Health check endpoint
   */
  http.get('*/agent/health', () => {
    return HttpResponse.json({ status: 'ok' });
  }),
];

/**
 * Create a handler for a specific scenario (useful for dynamic test configuration)
 */
export function createScenarioHandler(scenarioName: string) {
  return http.post('*/agent/run', async () => {
    const scenario = getScenario(scenarioName);
    await delay(10);
    return new HttpResponse(scenario.stream, {
      status: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
      },
    });
  });
}

/**
 * Get all available scenario names for tests
 */
export function getAvailableScenarios(): string[] {
  return Object.keys(ALL_SCENARIOS);
}

