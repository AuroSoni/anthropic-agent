"""CodeExecutionTool — async, sandbox-based code execution.

Migrated from anthropic_agent/common_tools/code_execution_tool.py.
Path-based file I/O replaced with sandbox.write_file() calls.
Inner tool function is now async.
"""
from __future__ import annotations

import inspect
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..tools.base import ConfigurableToolBase
from ..python_executors.local_python_executor import LocalPythonExecutor, CodeOutput
from ..python_executors.base import BASE_BUILTIN_MODULES, InterpreterError


class CodeExecutionTool(ConfigurableToolBase):
    """Stateful Python code execution tool with sandboxed imports and embedded tools.

    This tool provides notebook-like code execution where variables, imports, and
    function definitions persist across calls within the same session. It captures
    print outputs and last expression values, writing all output to agent-scoped files.

    The tool's docstring is dynamically generated to include:
    - Documentation for all embedded tools available in the code environment
    - List of authorized imports
    - Output truncation policy
    - Output file path pattern

    Example:
        >>> # Create tool with embedded calculator functions
        >>> @tool
        >>> def add(a: float, b: float) -> str:
        ...     '''Add two numbers.'''
        ...     return str(a + b)
        >>>
        >>> code_tool = CodeExecutionTool(
        ...     embedded_tools=[add],
        ...     authorized_imports=["numpy", "pandas"],
        ...     output_dir="code_runs",
        ...     max_output_chars=10_000,
        ... )
        >>>
        >>> # Register and attach sandbox
        >>> func = code_tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
        >>>
        >>> # Code can now call add() and use numpy/pandas
        >>> # Variables persist: x = 5 in one call, print(x) works in next call
    """

    DOCSTRING_TEMPLATE = """Execute Python code in a persistent environment.

This tool runs Python code with state preserved between calls (like a Jupyter notebook).
Variables, imports, and function definitions persist across executions.

**Available Functions:**
{embedded_tools_docs}

**Authorized Imports:**
{authorized_imports_str}

**Output Policy:**
- Maximum output characters: {max_output_chars}
- If output exceeds limit, only the tail is returned
- All output is written to: {output_path_pattern}

Args:
    code: Python code to execute. Can span multiple lines.

Returns:
    Execution output including:
    - Print statements captured during execution
    - Last expression value (if any)
    - Error messages (if execution failed)
    - Path to the output file

Example:
    ```python
    # First call - define variable
    x = 10
    print(f"x is {{x}}")
    ```

    ```python
    # Second call - variable persists
    y = x * 2
    print(f"y is {{y}}")
    ```
"""

    def __init__(
        self,
        output_dir: str = "code_runs",
        embedded_tools: List[Callable] | None = None,
        authorized_imports: List[str] | None = None,
        max_output_chars: int = 10_000,
        agent_uuid: str | None = None,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the CodeExecutionTool.

        Args:
            output_dir: Relative directory within the sandbox for writing output files.
                Output files are written to: <output_dir>/<agent_uuid>/<run_id>.txt
            embedded_tools: Optional list of @tool decorated functions that will be
                available to call from within executed code. Each tool's schema/docstring
                is included in the tool description.
            authorized_imports: Optional list of additional module names that can be
                imported. Merged with BASE_BUILTIN_MODULES (collections, datetime, etc.).
                Pass ["*"] to allow all imports (use with caution).
            max_output_chars: Maximum characters in output. If exceeded, only the tail
                is kept. Defaults to 10,000.
            agent_uuid: Optional agent UUID for output file paths. If not provided,
                must be injected later via set_agent_uuid() before execution.
            docstring_template: Optional custom docstring template with {placeholder} syntax.
            schema_override: Optional complete Anthropic tool schema dict for full control.
        """
        super().__init__(docstring_template=docstring_template, schema_override=schema_override)

        self.output_dir = output_dir
        self.embedded_tools = embedded_tools or []
        self.authorized_imports = authorized_imports or []
        self.max_output_chars = max_output_chars
        self.agent_uuid = agent_uuid

        # Build static_tools dict from embedded tools
        self._static_tools = self._build_static_tools()

        # Executor is created lazily to allow configuration changes
        self._executor: Optional[LocalPythonExecutor] = None

    def _build_static_tools(self) -> Dict[str, Callable]:
        """Build the static_tools dict from embedded_tools.

        Returns:
            Dict mapping tool names to callable functions.

        Raises:
            ValueError: If two embedded tools have the same name.
        """
        static_tools: Dict[str, Callable] = {}

        for tool_func in self.embedded_tools:
            # Get tool name from schema if available, else use __name__
            if hasattr(tool_func, "__tool_schema__"):
                name = tool_func.__tool_schema__.name
            else:
                name = tool_func.__name__

            if name in static_tools:
                raise ValueError(
                    f"Duplicate tool name '{name}' in embedded_tools. "
                    f"Each embedded tool must have a unique name."
                )

            static_tools[name] = tool_func

        return static_tools

    def _get_executor(self) -> LocalPythonExecutor:
        """Get or create the persistent LocalPythonExecutor.

        Returns:
            The executor instance, creating it on first call.
        """
        if self._executor is None:
            self._executor = LocalPythonExecutor(
                additional_authorized_imports=self.authorized_imports,
                max_print_output_length=self.max_output_chars,
            )
            # Send embedded tools to the executor
            self._executor.send_tools(self._static_tools)

        return self._executor

    def _format_embedded_tool_docs(self) -> str:
        """Format documentation for all embedded tools.

        Returns:
            Formatted string with tool name, description, and parameters for each tool.
        """
        if not self.embedded_tools:
            return "None - only built-in Python functions available."

        docs = []
        for tool_func in self.embedded_tools:
            if hasattr(tool_func, "__tool_schema__"):
                schema = tool_func.__tool_schema__
                name = schema.name
                description = schema.description

                # Format input parameters
                input_schema = schema.input_schema
                properties = input_schema.get("properties", {})
                required = set(input_schema.get("required", []))

                params = []
                for param_name, param_info in properties.items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    req_marker = "" if param_name in required else " (optional)"
                    params.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")

                params_str = "\n".join(params) if params else "    (no parameters)"
                docs.append(f"- `{name}`: {description}\n  Parameters:\n{params_str}")
            else:
                # Fallback to docstring
                name = tool_func.__name__
                docstring = inspect.getdoc(tool_func) or "No description available."
                # Take only first line of docstring for brevity
                first_line = docstring.split("\n")[0]
                docs.append(f"- `{name}`: {first_line}")

        return "\n".join(docs)

    def _format_authorized_imports(self) -> str:
        """Format the list of authorized imports.

        Returns:
            Formatted string listing all authorized imports.
        """
        all_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.authorized_imports))

        if "*" in self.authorized_imports:
            return "All imports are allowed (unrestricted mode)."

        return ", ".join(all_imports)

    def _get_template_context(self) -> Dict[str, Any]:
        """Return placeholder values for docstring template."""
        return {
            "embedded_tools_docs": self._format_embedded_tool_docs(),
            "authorized_imports_str": self._format_authorized_imports(),
            "max_output_chars": self.max_output_chars,
            "output_path_pattern": f"<output_dir>/<agent_uuid>/<run_id>.txt",
        }

    def set_agent_uuid(self, agent_uuid: str) -> None:
        """Set the agent UUID for output file paths.

        This method is called by the agent after UUID assignment to enable
        agent-scoped file output. It follows a duck-typed protocol so agents
        can call it on any tool that implements this method.

        Args:
            agent_uuid: The agent's UUID string.
        """
        self.agent_uuid = agent_uuid

    def reset_state(self) -> None:
        """Reset the executor state, clearing all variables and imports.

        Call this to start fresh without recreating the tool instance.
        """
        self._executor = None

    def _truncate_tail(self, content: str) -> str:
        """Truncate content to keep only the tail if it exceeds max_output_chars.

        Args:
            content: The content to potentially truncate.

        Returns:
            The content, possibly truncated with a header marker.
        """
        if len(content) <= self.max_output_chars:
            return content

        truncate_notice = f"\n... [truncated, showing last {self.max_output_chars} chars] ...\n"
        # Leave room for the notice
        tail_size = self.max_output_chars - len(truncate_notice)
        return truncate_notice + content[-tail_size:]

    async def _write_output_file(self, output: str) -> str:
        """Write output to an agent-scoped file via the sandbox.

        Args:
            output: The output content to write.

        Returns:
            The relative path to the written file within the sandbox.

        Raises:
            RuntimeError: If agent_uuid has not been set.
        """
        if self.agent_uuid is None:
            raise RuntimeError(
                "agent_uuid is not set. The agent must inject the UUID via set_agent_uuid() "
                "before code execution can write output files."
            )

        # Build relative path within sandbox
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = uuid.uuid4().hex[:8]
        filename = f"{timestamp}_{run_id}.txt"
        output_path = f"{self.output_dir}/{self.agent_uuid}/{filename}"

        # Write truncated output via sandbox (creates parent dirs automatically)
        truncated_output = self._truncate_tail(output)
        await self._sandbox.write_file(output_path, truncated_output)

        return output_path

    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with an Agent.

        Returns:
            A decorated code execution function that maintains state between calls.
            The docstring will reflect the configured embedded tools and imports.
            The function has a __tool_instance__ attribute for agent UUID injection.
        """
        instance = self

        async def code_execution(code: str) -> str:
            """Placeholder docstring - replaced by template."""
            executor = instance._get_executor()

            # Build output parts
            output_parts = []

            try:
                # Execute the code (executor is synchronous)
                code_output: CodeOutput = executor(code)

                # Add print outputs
                if code_output.logs:
                    output_parts.append(code_output.logs)

                # Add last expression value if present and not None
                if code_output.output is not None:
                    output_parts.append(f"\n[Last value]: {code_output.output}")

                # Note if this was a final answer
                if code_output.is_final_answer:
                    output_parts.append("\n[Final answer reached]")

            except InterpreterError as e:
                output_parts.append(f"[Execution Error]: {str(e)}")
            except Exception as e:
                output_parts.append(f"[Unexpected Error]: {type(e).__name__}: {str(e)}")

            # Combine output
            full_output = "".join(output_parts) if output_parts else "[No output]"

            # Write to file and get path
            try:
                output_path = await instance._write_output_file(full_output)
                path_line = f"output_path={output_path}"
            except RuntimeError as e:
                # UUID not set - include error but don't fail completely
                path_line = f"[Warning: {str(e)}]"

            # Truncate for return value
            truncated_output = instance._truncate_tail(full_output)

            return f"{path_line}\n\n{truncated_output}"

        func = self._apply_schema(code_execution)
        func.__tool_instance__ = instance
        return func
