/**
 * MSW server setup for Node.js test environment (vitest).
 */

import { setupServer } from 'msw/node';
import { handlers } from './handlers';

/**
 * MSW server instance for tests
 */
export const server = setupServer(...handlers);

/**
 * Setup function to be called in vitest setup file
 */
export function setupMswServer() {
  // Start server before all tests
  beforeAll(() => {
    server.listen({ onUnhandledRequest: 'warn' });
  });

  // Reset handlers after each test (removes any test-specific overrides)
  afterEach(() => {
    server.resetHandlers();
  });

  // Close server after all tests
  afterAll(() => {
    server.close();
  });
}

