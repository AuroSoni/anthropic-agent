# Background

The anthropic_agent/ framework is curerntly being refactored. Based on the issue that user describes, you have to investigate where it is arising from.
Based on your understanding, you'll have to dedice where to fix it:
- Option 1: Refactor the provider agnostic base (base_agent.py). This might need corressponding fixes in anthropic (agent.py) also.
- Option 2: Fix it in anthropic (agent.py, formatter.py, etc).
- Option 3: Fix it in openai (openai_agent.py, openai_formatters.py, etc).

## Reference

Anthropic SDK Docs: https://platform.claude.com/docs/en/get-started
OpenAI SDK Docs: https://platform.openai.com/docs/quickstart

## Key Principle

`base_agent.py` must remain **provider-agnostic**—no imports from `anthropic`, `openai`, or any provider SDK. And no data fields/ implementations that support only a particular provider. To understand the right point of fix you'd have to understand the design of the framework. 

## User Issue
Consider the following issue:

