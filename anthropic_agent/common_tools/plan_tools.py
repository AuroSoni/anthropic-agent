"""Plan mode tools for agent planning workflows.

This module provides four ConfigurableToolBase-based tools:
- EnterPlanModeTool: Frontend signal to enter plan mode (no backend logic)
- ExitPlanModeTool: Frontend signal to exit plan mode (no backend logic)
- CreatePlanTool: Create a new plan with title, overview, and todos (YAML persistence)
- EditPlanTool: Edit an existing plan using apply_patch hunk format

Plan state is persisted per agent as YAML files at:
    {base_dir}/{agent_uuid}/plans/{plan_id}.yaml
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml

from ..tools.base import ConfigurableToolBase
from ..tools.decorators import tool
from .apply_patch import ApplyPatchTool
from .todo_tool import _validate_todos


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLANS_DIR_NAME = "plans"


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------
def _slugify(text: str) -> str:
    """Convert text to a URL/filesystem-safe slug.

    Lowercase, replace non-alphanumeric characters with hyphens,
    collapse consecutive hyphens, strip leading/trailing hyphens.
    """
    slug = text.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    return slug


def _generate_plan_id(title: str) -> str:
    """Generate a plan ID from the title plus a short unique suffix."""
    slug = _slugify(title)
    suffix = uuid.uuid4().hex[:6]
    if not slug:
        return suffix
    return f"{slug}-{suffix}"


def _get_plans_dir(base_dir: Path, agent_uuid: str) -> Path:
    """Resolve the plans directory for an agent."""
    return base_dir / agent_uuid / PLANS_DIR_NAME


def _get_plan_file_path(base_dir: Path, agent_uuid: str, plan_id: str) -> Path:
    """Resolve the YAML file path for a specific plan."""
    return _get_plans_dir(base_dir, agent_uuid) / f"{plan_id}.yaml"


def _write_plan(
    base_dir: Path,
    agent_uuid: str,
    plan_id: str,
    title: str,
    overview: str,
    todos: list[dict[str, str]],
) -> Path:
    """Write a plan to a YAML file, creating directories as needed. Returns path."""
    path = _get_plan_file_path(base_dir, agent_uuid, plan_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "plan_id": plan_id,
        "title": title,
        "overview": overview,
        "todos": todos,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    return path


# ---------------------------------------------------------------------------
# EnterPlanModeTool
# ---------------------------------------------------------------------------
class EnterPlanModeTool(ConfigurableToolBase):
    """Frontend tool that signals entry into plan mode.

    This is a frontend-only tool — the function body is a no-op. The actual
    plan-mode UI transition is handled by the frontend when it receives
    this tool call.

    Example::

        tool = EnterPlanModeTool()
        enter_fn = tool.get_tool()
        agent = AnthropicAgent(tools=[enter_fn])
    """

    DOCSTRING_TEMPLATE = """Signal the frontend to enter plan mode.

Call this tool when you need to design an implementation approach before writing code.
Plan mode restricts your tool access to read-only exploration tools until the plan
is approved.

Use this before starting non-trivial implementation tasks to ensure alignment with
the user on the approach.

Returns:
    Confirmation that plan mode has been entered (handled by frontend).
"""

    def get_tool(self) -> Callable:
        """Return a @tool(executor='frontend') decorated function.

        Returns:
            A decorated enter_plan_mode function with frontend executor.
        """
        instance = self

        def enter_plan_mode() -> str:
            """Placeholder docstring - replaced by template."""
            pass

        enter_plan_mode.__doc__ = self._render_docstring()
        decorated = tool(enter_plan_mode, executor="frontend")
        decorated.__tool_instance__ = instance
        return decorated


# ---------------------------------------------------------------------------
# ExitPlanModeTool
# ---------------------------------------------------------------------------
class ExitPlanModeTool(ConfigurableToolBase):
    """Frontend tool that signals exit from plan mode.

    This is a frontend-only tool — the function body is a no-op. The actual
    plan-mode UI transition and plan review presentation is handled by
    the frontend when it receives this tool call.

    Example::

        tool = ExitPlanModeTool()
        exit_fn = tool.get_tool()
        agent = AnthropicAgent(tools=[exit_fn])
    """

    DOCSTRING_TEMPLATE = """Signal the frontend to exit plan mode and present the plan for review.

Call this tool after you have finished writing your plan. The frontend will present
the plan to the user for approval. On approval, you regain access to all tools
for implementation.

Args:
    plan_id: The ID of the plan to present for user review. This should be the
        plan_id returned by the create_plan tool.

Returns:
    The user's approval or rejection of the plan (handled by frontend).
"""

    def get_tool(self) -> Callable:
        """Return a @tool(executor='frontend') decorated function.

        Returns:
            A decorated exit_plan_mode function with frontend executor.
        """
        instance = self

        def exit_plan_mode(plan_id: str) -> str:
            """Placeholder docstring - replaced by template."""
            pass

        exit_plan_mode.__doc__ = self._render_docstring()
        decorated = tool(exit_plan_mode, executor="frontend")
        decorated.__tool_instance__ = instance
        return decorated


# ---------------------------------------------------------------------------
# CreatePlanTool
# ---------------------------------------------------------------------------
class CreatePlanTool(ConfigurableToolBase):
    """Configurable tool for creating a new plan with title, overview, and todos.

    Plans are persisted as YAML files scoped to the agent UUID at:
        {base_dir}/{agent_uuid}/plans/{plan_id}.yaml

    Example::

        tool = CreatePlanTool(base_dir="/data/plans")
        tool.set_agent_uuid("my-agent-uuid")
        create_fn = tool.get_tool()
        agent = AnthropicAgent(tools=[create_fn])
    """

    DOCSTRING_TEMPLATE = """Create a new plan with a title, overview, and todo items.

The plan is persisted as a YAML file at {storage_path_pattern}.
Returns the generated plan_id which is used to reference the plan in other tools.

Args:
    title: A short descriptive title for the plan (e.g., "Refactor Auth System").
        Used to generate the plan_id slug.
    overview: A detailed plan overview in markdown format. Describe the goals,
        approach, key decisions, and any constraints.
    todos: A YAML-formatted list of todo items. Each item must have these fields:
        - id (string): Short unique identifier (e.g., "research", "implement")
        - content (string): Task description in imperative form (e.g., "Research OAuth2 libraries")
        - status (string): One of "pending", "in_progress", or "completed"
        - activeForm (string): Present-continuous description (e.g., "Researching OAuth2 libraries")

        Example YAML:
        - id: research
          content: Research OAuth2 libraries
          status: pending
          activeForm: Researching OAuth2 libraries
        - id: implement
          content: Implement OAuth2 flow
          status: pending
          activeForm: Implementing OAuth2 flow

Returns:
    The generated plan_id string (e.g., "refactor-auth-system-a1b2c3") on success,
    or an error message with guidance if validation fails.
"""

    def __init__(
        self,
        base_dir: str | Path,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the CreatePlanTool.

        Args:
            base_dir: Root directory for plan YAML files. Each agent gets
                a subdirectory: {base_dir}/{agent_uuid}/plans/{plan_id}.yaml
            docstring_template: Optional custom docstring template with
                {placeholder} syntax. Available: {storage_path_pattern}.
            schema_override: Optional complete Anthropic tool schema dict.
        """
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )
        self.base_dir = Path(base_dir)
        self.agent_uuid: str | None = None

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "storage_path_pattern": (
                f"{self.base_dir}/<agent_uuid>/{PLANS_DIR_NAME}/<plan_id>.yaml"
            ),
        }

    def set_agent_uuid(self, agent_uuid: str) -> None:
        """Set the agent UUID for scoping the plan files.

        Called by AnthropicAgent._inject_agent_uuid_to_tools() via the
        ``__tool_instance__`` / ``set_agent_uuid`` duck-typed protocol.
        """
        self.agent_uuid = agent_uuid

    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.

        Returns:
            A decorated create_plan function. The function has a
            ``__tool_instance__`` attribute for agent UUID injection.
        """
        instance = self

        def create_plan(title: str, overview: str, todos: str) -> str:
            """Placeholder docstring - replaced by template."""
            if instance.agent_uuid is None:
                return (
                    "Error: agent_uuid is not set. Cannot create plan "
                    "without an agent session."
                )

            if not title.strip():
                return "Error: Plan title cannot be empty."

            # Parse YAML todos
            try:
                todo_list = yaml.safe_load(todos)
            except yaml.YAMLError as e:
                return (
                    f"Error: Failed to parse todos YAML: {e}. "
                    f"Please check YAML syntax — todos must be a YAML list "
                    f"of objects with id, content, status, and activeForm fields."
                )

            if todo_list is None:
                todo_list = []

            if not isinstance(todo_list, list):
                return (
                    "Error: Todos must be a YAML list of objects. "
                    "Got a single value instead."
                )

            # Validate using the same rules as TodoWriteTool
            error = _validate_todos(todo_list)
            if error:
                return error

            # Strip to required keys only
            clean_todos = [
                {k: todo[k] for k in ("id", "content", "status", "activeForm")}
                for todo in todo_list
            ]

            # Generate plan ID
            plan_id = _generate_plan_id(title.strip())

            # Write plan YAML
            try:
                _write_plan(
                    instance.base_dir,
                    instance.agent_uuid,
                    plan_id,
                    title.strip(),
                    overview,
                    clean_todos,
                )
            except Exception as e:
                return f"Error: Failed to write plan: {e}"

            return plan_id

        create_plan.__tool_instance__ = instance
        return self._apply_schema(create_plan)


# ---------------------------------------------------------------------------
# EditPlanTool
# ---------------------------------------------------------------------------
class EditPlanTool(ConfigurableToolBase):
    """Configurable tool for editing an existing plan using apply_patch hunks.

    Accepts hunk content (without file headers) for editing the plan overview
    and/or todos. The tool wraps the hunks with the correct apply_patch file
    headers and applies them to the plan's YAML file.

    Example::

        tool = EditPlanTool(base_dir="/data/plans")
        tool.set_agent_uuid("my-agent-uuid")
        edit_fn = tool.get_tool()
        agent = AnthropicAgent(tools=[edit_fn])
    """

    DOCSTRING_TEMPLATE = """Edit an existing plan by applying patch hunks to its YAML file.

Provide hunk content (without file headers) to modify the plan overview and/or
todos sections. The tool wraps hunks with the correct apply_patch file headers
internally.

Plans are stored at {storage_path_pattern}.

**Hunk Format:**
Use the same hunk format as the apply_patch tool, but without the
``*** Begin Patch`` / ``*** Update File:`` / ``*** End Patch`` wrappers.

Example hunk to update the overview:
```
@@ overview: |
   ## Old Title
-  Old description text
+  New description text
```

Example hunk to update todos:
```
@@
 todos:
   - id: research
-    status: pending
+    status: completed
```

Args:
    plan_id: The ID of the plan to edit (as returned by create_plan).
    overview_patch: Hunk content to apply to the plan overview section.
        Omit or pass empty string if not editing the overview.
    todos_patch: Hunk content to apply to the plan todos section.
        Omit or pass empty string if not editing the todos.

Returns:
    Confirmation of what was patched on success, or an error message
    with details if the patch fails. Re-read the plan file to get
    current content if a context mismatch occurs.
"""

    def __init__(
        self,
        base_dir: str | Path,
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        """Initialize the EditPlanTool.

        Args:
            base_dir: Root directory for plan YAML files. Must match the
                base_dir used by the corresponding CreatePlanTool.
            docstring_template: Optional custom docstring template with
                {placeholder} syntax. Available: {storage_path_pattern}.
            schema_override: Optional complete Anthropic tool schema dict.
        """
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )
        self.base_dir = Path(base_dir)
        self.agent_uuid: str | None = None

    def _get_template_context(self) -> Dict[str, Any]:
        return {
            "storage_path_pattern": (
                f"{self.base_dir}/<agent_uuid>/{PLANS_DIR_NAME}/<plan_id>.yaml"
            ),
        }

    def set_agent_uuid(self, agent_uuid: str) -> None:
        """Set the agent UUID for scoping the plan files.

        Called by AnthropicAgent._inject_agent_uuid_to_tools() via the
        ``__tool_instance__`` / ``set_agent_uuid`` duck-typed protocol.
        """
        self.agent_uuid = agent_uuid

    def get_tool(self) -> Callable:
        """Return a @tool decorated function for use with AnthropicAgent.

        Returns:
            A decorated edit_plan function. The function has a
            ``__tool_instance__`` attribute for agent UUID injection.
        """
        instance = self

        def edit_plan(
            plan_id: str,
            overview_patch: str = "",
            todos_patch: str = "",
        ) -> str:
            """Placeholder docstring - replaced by template."""
            if instance.agent_uuid is None:
                return (
                    "Error: agent_uuid is not set. Cannot edit plan "
                    "without an agent session."
                )

            # Verify the plan file exists
            plan_path = _get_plan_file_path(
                instance.base_dir, instance.agent_uuid, plan_id
            )
            if not plan_path.exists():
                return (
                    f"Error: Plan '{plan_id}' not found. "
                    f"Expected file at: {plan_path}"
                )

            # Determine which patches to apply
            patches_to_apply: list[tuple[str, str]] = []
            if overview_patch.strip():
                patches_to_apply.append(("overview", overview_patch))
            if todos_patch.strip():
                patches_to_apply.append(("todos", todos_patch))

            if not patches_to_apply:
                return (
                    "Error: No patches provided. Supply at least one of "
                    "overview_patch or todos_patch."
                )

            # Apply patches sequentially
            plans_dir = _get_plans_dir(instance.base_dir, instance.agent_uuid)
            filename = f"{plan_id}.yaml"
            applied = []

            for label, hunk_content in patches_to_apply:
                full_patch = (
                    f"*** Begin Patch\n"
                    f"*** Update File: {filename}\n"
                    f"{hunk_content.rstrip()}\n"
                    f"*** End Patch"
                )

                patch_tool = ApplyPatchTool(
                    base_path=plans_dir,
                    enforce_allowlist=False,
                )
                apply_fn = patch_tool.get_tool()
                result = apply_fn(patch=full_patch)

                # Check for errors in the JSON result
                try:
                    result_data = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    return f"Error applying {label} patch: {result}"

                if result_data.get("status") == "error":
                    error_msg = result_data.get("error", "Unknown error")
                    hint = result_data.get("hint", "")
                    msg = f"Error applying {label} patch: {error_msg}"
                    if hint:
                        msg += f"\nHint: {hint}"
                    return msg

                applied.append(label)

            return f"Plan '{plan_id}' updated successfully. Patched: {', '.join(applied)}."

        edit_plan.__tool_instance__ = instance
        return self._apply_schema(edit_plan)
