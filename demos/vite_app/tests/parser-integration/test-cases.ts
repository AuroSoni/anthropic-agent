/**
 * Test case definitions for parser integration tests.
 * Ported from demos/fastapi_server/agent_api_test.ipynb
 * 
 * NOTE: Image tests use read_image_raw tool which reads from a relative path.
 * Ensure the test image exists at the expected path relative to fastapi_server.
 */
import type { TestCase } from './types';

// =============================================================================
// Prompts
// =============================================================================

const PROMPTS = {
  // Basic prompts
  whatIsAi: "Hi, what is AI?",
  mathExpression: "Hi, what is 5*(3+2/4)?",
  ycombinator: "Fetch me the latest happenings in the Y Combinator ecosystem",
  markdownReport: "Produce a markdown report with a table + code block + nested bullets about 'How to reduce latency in an API.'",
  fileGeneration: "Generate a JSON file named `users.json` with 20 synthetic users: id, name, email, country.",
  
  // Image tool result prompt (triggers read_image_raw tool)
  imageToolResult: "Read and analyze the image at ../../currency_receipt_usd_jpy.png",
  
  // PDF citations prompt
  pdfCitations: [
    {
      type: "document",
      source: {
        type: "url",
        url: "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf",
      },
      citations: { enabled: true },
    },
    {
      type: "text",
      text: "What are the key findings in this document?",
    },
  ],
  
  // Content citations prompt
  contentCitations: {
    role: "user",
    content: [
      {
        type: "document",
        source: {
          type: "text",
          media_type: "text/plain",
          data: "The grass is green. The sky is blue.",
        },
        title: "My Document",
        context: "This is a trustworthy document.",
        citations: { enabled: true },
      },
      {
        type: "text",
        text: "What color is the grass and sky?",
      },
    ],
  },
  
  // Frontend tools prompt (triggers user_confirm)
  frontendToolsConfirm: "Calculate 25 * 4 for me. Confirm with user before proceeding.",
  
  // Chart embedded tags prompt
  chartEmbedded: `
<chart_config>
**What Is It?**
A **Chart.js JSON config** is a text format that describes how your chart should look and what data it displays. Think of it like a recipe written in a specific format that Chart.js can read. The chart js json config can be used to show charts in the response.

**Basic Structure**:
\`\`\`json
{
  "type": "bar",           // Chart type: 'bar', 'line', 'pie', 'doughnut', etc.
  "data": {                // Your chart data
    "labels": [...],       // X-axis labels (array of strings)
    "datasets": [...]      // Your data series (array of objects)
  },
  "options": {             // How the chart looks and behaves
    ...
  }
}
\`\`\`

To show charts, embed Chart.js JSON config between \`<chart>\` tags.

**Important:** Place the JSON directly inside the tags without escaping or quotes:

\`\`\`
<chart>
{
  "type": "bar",
  "data": {
    "labels": ["Label 1", "Label 2"],
    "datasets": [{"label": "My Data", "data": [10, 20]}]
  }
}
</chart>
\`\`\`

**Complete Template**:
\`\`\`json
{
  "type": "bar",
  "data": {
    "labels": ["Label 1", "Label 2", "Label 3"],
    "datasets": [{
      "label": "My Dataset",
      "data": [10, 20, 30],
      "backgroundColor": "rgba(75, 192, 192, 0.2)",
      "borderColor": "rgba(75, 192, 192, 1)",
      "borderWidth": 1
    }]
  },
  "options": {
    "responsive": true,
    "plugins": {
      "title": {
        "display": true,
        "text": "My Chart Title"
      }
    },
    "scales": {
      "y": {
        "beginAtZero": true
      }
    }
  }
}
\`\`\`

**Rules**:
✅ **CAN include**: Numbers, Strings, Booleans, Arrays, Objects, Null

❌ **CANNOT include**:
- Functions or code or callbacks (config is serialized)
- Variables (like \`myVariable\`)
- Comments (remove \`//\` when saving as JSON)
- Single quotes (use double quotes only)
</chart_config>

<user_prompt>
Present world population demographic data in a chart.
</user_prompt>
`,
};

// =============================================================================
// Test Cases
// =============================================================================

export const TEST_CASES: TestCase[] = [
  // No tools
  {
    name: "no_tools_what_is_ai",
    prompt: PROMPTS.whatIsAi,
    agentType: "agent_no_tools",
    description: "Basic text response without tools",
  },
  
  // Client tools
  {
    name: "client_tools_math",
    prompt: PROMPTS.mathExpression,
    agentType: "agent_client_tools",
    description: "Client-side tool call for math calculation",
  },
  
  // Server tools - Raw format
  {
    name: "server_tools_math_raw",
    prompt: PROMPTS.mathExpression,
    agentType: "agent_all_raw",
    description: "Server tools with raw format - math calculation",
  },
  
  // Server tools - XML format
  {
    name: "server_tools_math_xml",
    prompt: PROMPTS.mathExpression,
    agentType: "agent_all_xml",
    description: "Server tools with XML format - math calculation",
  },
  
  // Y Combinator - Raw format
  {
    name: "ycombinator_raw",
    prompt: PROMPTS.ycombinator,
    agentType: "agent_all_raw",
    description: "Web search tool with raw format",
  },
  
  // Y Combinator - XML format
  {
    name: "ycombinator_xml",
    prompt: PROMPTS.ycombinator,
    agentType: "agent_all_xml",
    description: "Web search tool with XML format",
  },
  
  // PDF citations - Raw format
  {
    name: "pdf_citations_raw",
    prompt: PROMPTS.pdfCitations,
    agentType: "agent_all_raw",
    description: "PDF document citations with raw format",
  },
  
  // PDF citations - XML format
  {
    name: "pdf_citations_xml",
    prompt: PROMPTS.pdfCitations,
    agentType: "agent_all_xml",
    description: "PDF document citations with XML format",
  },
  
  // Content citations - Raw format
  {
    name: "content_citations_raw",
    prompt: PROMPTS.contentCitations,
    agentType: "agent_all_raw",
    description: "Text content citations with raw format",
  },
  
  // Content citations - XML format
  {
    name: "content_citations_xml",
    prompt: PROMPTS.contentCitations,
    agentType: "agent_all_xml",
    description: "Text content citations with XML format",
  },
  
  // Markdown report - Raw format
  {
    name: "markdown_report_raw",
    prompt: PROMPTS.markdownReport,
    agentType: "agent_all_raw",
    description: "Complex markdown rendering with raw format",
  },
  
  // Markdown report - XML format
  {
    name: "markdown_report_xml",
    prompt: PROMPTS.markdownReport,
    agentType: "agent_all_xml",
    description: "Complex markdown rendering with XML format",
  },
  
  // File generation - Raw format
  {
    name: "file_generation_raw",
    prompt: PROMPTS.fileGeneration,
    agentType: "agent_all_raw",
    description: "File creation with meta_files - raw format",
  },
  
  // File generation - XML format
  {
    name: "file_generation_xml",
    prompt: PROMPTS.fileGeneration,
    agentType: "agent_all_xml",
    description: "File creation with meta_files - XML format",
  },
  
  // Chart embedded - Raw format
  {
    name: "chart_embedded_raw",
    prompt: PROMPTS.chartEmbedded,
    agentType: "agent_all_raw",
    description: "Embedded <chart> tags with raw format",
  },
  
  // Chart embedded - XML format
  {
    name: "chart_embedded_xml",
    prompt: PROMPTS.chartEmbedded,
    agentType: "agent_all_xml",
    description: "Embedded <chart> tags with XML format",
  },
  
  // Frontend tools - XML format (pauses for user_confirm)
  {
    name: "frontend_tools_confirm_xml",
    prompt: PROMPTS.frontendToolsConfirm,
    agentType: "agent_frontend_tools",
    description: "Frontend tools with user_confirm - XML format, pauses at awaiting_frontend_tools",
  },
  
  // Frontend tools - Raw format (pauses for user_confirm)
  {
    name: "frontend_tools_confirm_raw",
    prompt: PROMPTS.frontendToolsConfirm,
    agentType: "agent_frontend_tools_raw",
    description: "Frontend tools with user_confirm - raw format, pauses at awaiting_frontend_tools",
  },
  
  // Image tool result - Raw format
  {
    name: "image_tool_result_raw",
    prompt: PROMPTS.imageToolResult,
    agentType: "agent_all_raw",
    description: "Tool result with image - raw format, tests read_image_raw tool",
  },
  
  // Image tool result - XML format
  {
    name: "image_tool_result_xml",
    prompt: PROMPTS.imageToolResult,
    agentType: "agent_all_xml",
    description: "Tool result with image - XML format, tests read_image_raw tool",
  },
];

/**
 * Get test cases filtered by name pattern.
 */
export function getTestCases(filter?: string): TestCase[] {
  if (!filter) return TEST_CASES;
  return TEST_CASES.filter(tc => tc.name.includes(filter));
}

/**
 * Get a single test case by exact name.
 */
export function getTestCase(name: string): TestCase | undefined {
  return TEST_CASES.find(tc => tc.name === name);
}

