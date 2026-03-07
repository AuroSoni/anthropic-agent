"""Plan mode tools — async, sandbox-based planning workflows.

Migrated from anthropic_agent/common_tools/plan_tools.py.
YAML persistence goes through sandbox read_file/write_file.
Agent UUID scoping removed — sandbox provides isolation.
"""
from __future__ import annotations

import json
import re
import uuid
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
    """Convert text to a URL/filesystem-safe slug."""
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


# ---------------------------------------------------------------------------
# EnterPlanModeTool
# ---------------------------------------------------------------------------
class EnterPlanModeTool(ConfigurableToolBase):
    """Frontend tool that signals entry into plan mode.

    This is a frontend-only tool — the function body is a no-op.

    Example:
        >>> tool = EnterPlanModeTool()
        >>> func = tool.get_tool()
        >>> registry.register_tools([func])
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

    This is a frontend-only tool — the function body is a no-op.

    Example:
        >>> tool = ExitPlanModeTool()
        >>> func = tool.get_tool()
        >>> registry.register_tools([func])
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
    """Configurable tool for creating a new plan with sandbox-based persistence.

    Plans are stored at plans/{plan_id}.yaml within the sandbox.

    Example:
        >>> tool = CreatePlanTool()
        >>> func = tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """Create a new plan with a title, overview, and todo items.

The plan is persisted as a YAML file at plans/<plan_id>.yaml.
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
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )

    def get_tool(self) -> Callable:
        instance = self

        async def create_plan(title: str, overview: str, todos: str) -> str:
            """Placeholder docstring - replaced by template."""
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

            error = _validate_todos(todo_list)
            if error:
                return error

            clean_todos = [
                {k: todo[k] for k in ("id", "content", "status", "activeForm")}
                for todo in todo_list
            ]

            plan_id = _generate_plan_id(title.strip())

            # Write plan YAML via sandbox
            try:
                data = {
                    "plan_id": plan_id,
                    "title": title.strip(),
                    "overview": overview,
                    "todos": clean_todos,
                }
                content = yaml.dump(
                    data,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
                plan_path = f"{PLANS_DIR_NAME}/{plan_id}.yaml"
                await instance._sandbox.write_file(plan_path, content)
            except Exception as e:
                return f"Error: Failed to write plan: {e}"

            return plan_id

        func = self._apply_schema(create_plan)
        func.__tool_instance__ = instance
        return func


# ---------------------------------------------------------------------------
# EditPlanTool
# ---------------------------------------------------------------------------
class EditPlanTool(ConfigurableToolBase):
    """Configurable tool for editing an existing plan using apply_patch hunks.

    Shares the parent's sandbox with an internal ApplyPatchTool instance.

    Example:
        >>> tool = EditPlanTool()
        >>> func = tool.get_tool()
        >>> registry.register_tools([func])
        >>> registry.attach_sandbox(sandbox)
    """

    DOCSTRING_TEMPLATE = """Edit an existing plan by applying patch hunks to its YAML file.

Provide hunk content (without file headers) to modify the plan overview and/or
todos sections. The tool wraps hunks with the correct apply_patch file headers
internally.

Plans are stored at plans/<plan_id>.yaml.

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
        docstring_template: str | None = None,
        schema_override: dict | None = None,
    ):
        super().__init__(
            docstring_template=docstring_template,
            schema_override=schema_override,
        )

    def get_tool(self) -> Callable:
        instance = self

        async def edit_plan(
            plan_id: str,
            overview_patch: str = "",
            todos_patch: str = "",
        ) -> str:
            """Placeholder docstring - replaced by template."""
            # Verify the plan file exists
            plan_path = f"{PLANS_DIR_NAME}/{plan_id}.yaml"
            exists = await instance._sandbox.file_exists(plan_path)
            if not exists:
                return f"Error: Plan '{plan_id}' not found at {plan_path}."

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

            # Create an ApplyPatchTool that shares our sandbox
            patch_tool = ApplyPatchTool()
            patch_tool.set_sandbox(instance._sandbox)
            apply_fn = patch_tool.get_tool()

            # Apply patches sequentially
            filename = f"{PLANS_DIR_NAME}/{plan_id}.yaml"
            applied = []

            for label, hunk_content in patches_to_apply:
                full_patch = (
                    f"*** Begin Patch\n"
                    f"*** Update File: {filename}\n"
                    f"{hunk_content.rstrip()}\n"
                    f"*** End Patch"
                )

                result = await apply_fn(patch=full_patch)

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

        func = self._apply_schema(edit_plan)
        func.__tool_instance__ = instance
        return func
