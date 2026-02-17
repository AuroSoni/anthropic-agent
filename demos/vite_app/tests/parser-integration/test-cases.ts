/**
 * Test case definitions for parser integration tests (JSON format only).
 * Ported from demos/fastapi_server/agent_api_test.ipynb
 *
 * NOTE: Image tests use read_image_raw tool which reads from a relative path.
 * Ensure the test image exists at the expected path relative to fastapi_server.
 */
import type { TestCase } from './types';

// =============================================================================
// File URLs for test fixtures
// =============================================================================

/** Real receipt image (relative to repo root, resolved by the test runner) */
const IMAGE_PATH = '../../../../currency_receipt_usd_jpy.png';

/** Anthropic Claude model card PDF */
const PDF_URL =
  'https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf';

/** Simple CSV for unsupported file type test (inline is fine — it's text, not arbitrary binary) */
const CSV_B64 = btoa('name,age\nAlice,30\nBob,25\n');

// =============================================================================
// Prompts
// =============================================================================

const PROMPTS = {
  // Basic prompts
  whatIsAi: 'Hi, what is AI?',
  mathExpression: 'Hi, what is 5*(3+2/4)?',
  ycombinator: 'Fetch me the latest happenings in the Y Combinator ecosystem',
  markdownReport:
    "Produce a markdown report with a table + code block + nested bullets about 'How to reduce latency in an API.'",
  fileGeneration:
    'Generate a JSON file named `users.json` with 20 synthetic users: id, name, email, country.',

  // Image tool result prompt (triggers read_image_raw tool)
  imageToolResult: 'Read and analyze the image at ../../currency_receipt_usd_jpy.png',

  // Inline base64 image in prompt (image data resolved at runtime by the test runner)
  imageBase64InPrompt: 'Describe this image briefly.',

  // PDF citations prompt
  pdfCitations: [
    {
      type: 'document',
      source: {
        type: 'url',
        url: 'https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf',
      },
      citations: { enabled: true },
    },
    {
      type: 'text',
      text: 'What are the key findings in this document?',
    },
  ],

  // Content citations prompt
  contentCitations: {
    role: 'user',
    content: [
      {
        type: 'document',
        source: {
          type: 'text',
          media_type: 'text/plain',
          data: 'The grass is green. The sky is blue.',
        },
        title: 'My Document',
        context: 'This is a trustworthy document.',
        citations: { enabled: true },
      },
      {
        type: 'text',
        text: 'What color is the grass and sky?',
      },
    ],
  },

  // Frontend tools prompt (triggers user_confirm)
  frontendToolsConfirm: 'Calculate 25 * 4 for me. Confirm with user before proceeding.',

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

  // Literal newlines / backslash behavior
  literalNewlines:
    'Explain the behavior of backslash (\\) in a string. Also explain what \\n does. How is it different from \\\\n or \\r\\n?',

  // Multipart prompts (text portion only; files are in TestCase.files)
  multipartImageUpload: 'I paid the receipt in USD. What is the amount in INR?',
  multipartPdfUpload: 'What are the key findings in this document?',
  multipartTextOnly: 'Hello, what is 2 + 2?',
  multipartUnsupportedFile: 'Analyze this data',

  // Anthropic Skills prompts
  skillsXlsx:
    'Create an Excel spreadsheet with a simple monthly budget: categories (Rent, Food, Transport, Entertainment, Savings) and amounts for 3 months.',
  skillsPptx:
    'Create a PowerPoint presentation with 3 slides about the solar system: title slide, inner planets overview, and outer planets overview.',
};

// =============================================================================
// Test Cases
// =============================================================================

export const TEST_CASES: TestCase[] = [
  // --- No tools ---
  {
    name: 'no_tools_what_is_ai',
    prompt: PROMPTS.whatIsAi,
    agentType: 'agent_no_tools',
    description: 'Basic text response without tools',
  },

  // --- Client tools ---
  {
    name: 'client_tools_math',
    prompt: PROMPTS.mathExpression,
    agentType: 'agent_client_tools',
    description: 'Client-side tool call for math calculation',
  },

  // --- Server tools ---
  {
    name: 'server_tools_math',
    prompt: PROMPTS.mathExpression,
    agentType: 'agent_all_json',
    description: 'Server tools with JSON format - math calculation',
  },

  // --- Y Combinator (web search) ---
  {
    name: 'ycombinator',
    prompt: PROMPTS.ycombinator,
    agentType: 'agent_all_json',
    description: 'Web search tool with JSON format',
  },

  // --- PDF citations ---
  {
    name: 'pdf_citations',
    prompt: PROMPTS.pdfCitations,
    agentType: 'agent_all_json',
    description: 'PDF document citations',
  },

  // --- Content citations ---
  {
    name: 'content_citations',
    prompt: PROMPTS.contentCitations,
    agentType: 'agent_all_json',
    description: 'Text content citations',
  },

  // --- Markdown report ---
  {
    name: 'markdown_report',
    prompt: PROMPTS.markdownReport,
    agentType: 'agent_all_json',
    description: 'Complex markdown rendering (table + code + bullets)',
  },

  // --- File generation ---
  {
    name: 'file_generation',
    prompt: PROMPTS.fileGeneration,
    agentType: 'agent_all_json',
    description: 'File creation with meta_files',
  },

  // --- Chart embedded tags ---
  {
    name: 'chart_embedded',
    prompt: PROMPTS.chartEmbedded,
    agentType: 'agent_all_json',
    description: 'Embedded <chart> tags in response',
  },

  // --- Frontend tools (user_confirm) ---
  {
    name: 'frontend_tools_confirm',
    prompt: PROMPTS.frontendToolsConfirm,
    agentType: 'agent_frontend_tools',
    description: 'Frontend tools with user_confirm - pauses at awaiting_frontend_tools',
  },

  // --- Image tool result ---
  {
    name: 'image_tool_result',
    prompt: PROMPTS.imageToolResult,
    agentType: 'agent_all_json',
    description: 'Tool result with image - tests read_image_raw tool',
  },

  // --- NEW: Inline base64 image in prompt (image fetched and encoded at runtime) ---
  {
    name: 'image_base64_in_prompt',
    prompt: PROMPTS.imageBase64InPrompt,
    agentType: 'agent_all_json',
    endpoint: 'multipart',
    files: [{ filename: 'currency_receipt_usd_jpy.png', mimeType: 'image/png', url: IMAGE_PATH }],
    description: 'Real image uploaded via multipart - tests image content in prompt',
  },

  // --- NEW: Literal newlines / backslash ---
  {
    name: 'literal_newlines_backslash',
    prompt: PROMPTS.literalNewlines,
    agentType: 'agent_all_json',
    description: 'Backslash and newline character rendering in streamed text',
  },

  // --- NEW: Multipart image upload ---
  {
    name: 'multipart_image_upload',
    prompt: PROMPTS.multipartImageUpload,
    agentType: 'agent_all_json',
    endpoint: 'multipart',
    files: [{ filename: 'currency_receipt_usd_jpy.png', mimeType: 'image/png', url: IMAGE_PATH }],
    description: 'Image file uploaded via multipart/form-data endpoint',
  },

  // --- NEW: Multipart PDF upload ---
  {
    name: 'multipart_pdf_upload',
    prompt: PROMPTS.multipartPdfUpload,
    agentType: 'agent_all_json',
    endpoint: 'multipart',
    files: [{ filename: 'Claude-3-Model-Card.pdf', mimeType: 'application/pdf', url: PDF_URL }],
    description: 'PDF file uploaded via multipart/form-data endpoint',
  },

  // --- NEW: Multipart text only (no files) ---
  {
    name: 'multipart_text_only',
    prompt: PROMPTS.multipartTextOnly,
    agentType: 'agent_all_json',
    endpoint: 'multipart',
    description: 'Multipart endpoint with text only, no file attachments',
  },

  // --- NEW: Multipart unsupported file type (expect 400) ---
  {
    name: 'multipart_unsupported_file',
    prompt: PROMPTS.multipartUnsupportedFile,
    agentType: 'agent_all_json',
    endpoint: 'multipart',
    files: [{ filename: 'data.csv', mimeType: 'text/csv', content: CSV_B64 }],
    expectError: 400,
    description: 'Unsupported file type rejected with HTTP 400',
  },

  // --- NEW: Skills - Excel spreadsheet ---
  {
    name: 'skills_xlsx',
    prompt: PROMPTS.skillsXlsx,
    agentType: 'agent_skills',
    description: 'Anthropic Skills - Excel spreadsheet generation via code_execution',
  },

  // --- NEW: Skills - PowerPoint presentation ---
  {
    name: 'skills_pptx',
    prompt: PROMPTS.skillsPptx,
    agentType: 'agent_skills',
    description: 'Anthropic Skills - PowerPoint presentation generation via code_execution',
  },
];

/**
 * Get test cases filtered by name pattern.
 */
export function getTestCases(filter?: string): TestCase[] {
  if (!filter) return TEST_CASES;
  return TEST_CASES.filter((tc) => tc.name.includes(filter));
}

/**
 * Get a single test case by exact name.
 */
export function getTestCase(name: string): TestCase | undefined {
  return TEST_CASES.find((tc) => tc.name === name);
}
