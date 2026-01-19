from importlib.util import find_spec
import logging
from dataclasses import dataclass
import ast
from typing import Any, Callable
from .base import truncate_content, BASE_BUILTIN_MODULES, InterpreterError, FinalAnswerException, PrintContainer, BASE_PYTHON_TOOLS, DEFAULT_MAX_LEN_OUTPUT
from .ast_evaluator import evaluate_ast

logger = logging.getLogger(__name__)

def evaluate_python_code(
    code: str,
    static_tools: dict[str, callable] | None = None,
    custom_tools: dict[str, callable] | None = None,
    state: dict[str, Any] | None = None,
    authorized_imports: list[str] | None = None,
    max_print_output_length: int | None = None,
):
    """
    Evaluate a python expression using the content of the variables stored in a state and only evaluating a given set
    of functions.

    This function will recurse through the nodes of the tree provided.

    Args:
        code (`str`):
            The code to evaluate.
        static_tools (`Dict[str, callable]`):
            The functions that may be called during the evaluation. These can also be agents in a multiagent setting.
            These tools cannot be overwritten in the code: any assignment to their name will raise an error.
        custom_tools (`Dict[str, callable]`):
            The functions that may be called during the evaluation.
            These tools can be overwritten in the code: any assignment to their name will overwrite them.
        state (`Dict[str, Any]`):
            A dictionary mapping variable names to values. The `state` should contain the initial inputs but will be
            updated by this function to contain all variables as they are evaluated.
            The print outputs will be stored in the state under the key "_print_outputs".
    """
    # First, check if the code is a valid Python expression
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}\n"
            f"{e.text}"
            f"{' ' * (e.offset or 0)}^\n"
            f"Error: {str(e)}"
        )

    if state is None:
        state = {}

    static_tools = static_tools.copy() if static_tools is not None else {}
    custom_tools = custom_tools.copy() if custom_tools is not None else {}
    
    result = None
    state["_print_outputs"] = PrintContainer()
    state["_operations_count"] = {"counter" : 0}
    
    if "final_answer" in static_tools:
        # Patch final_answer to raise an exception instead of returning a value
        # This is to ensure that the code flow is interrrupted as soon as the final asnwer is seen.
        # This will not happen in normal execution flow.
        previous_final_answer = static_tools["final_answer"]
        def final_answer(*args, **kwargs):
            raise FinalAnswerException(previous_final_answer(*args, **kwargs))
        static_tools["final_answer"] = final_answer
    
    # Now start the actual execution
    try:
        for node in expression.body:
            result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]),
            max_print_output_length,
        )
        
        is_final_answer = False
        return result, is_final_answer
    
    except FinalAnswerException as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]),
            max_print_output_length,
        )
        is_final_answer = True
        return e.value, is_final_answer
    except Exception as e:
        state["_print_outputs"].value = truncate_content(
            str(state["_print_outputs"]),
            max_print_output_length,
        )
        raise InterpreterError(
            f"Code execution failed at line '{ast.get_source_segment(code, node)}' due to {type(e).__name__}: {e}"
        )

@dataclass
class CodeOutput:
    """
    A dataclass to store the output of a code execution from a single code action.
    
    Args:
        output (`Any`):
            The output of the current code execution step.
        logs (`str`):
            The logs of the entire agent run including the current code execution step.
        is_final_answer (`bool`):
            Whether the current code execution step has the final answer.
    """
    output: Any
    logs: str
    is_final_answer: bool

class PythonExecutor:
    """
    Base class for all Python executors.
    """
    pass

class LocalPythonExecutor(PythonExecutor):
    """
    Executor of Python code in a local environment.

    This executor evaluates Python code with restricted access to imports and built-in functions,
    making it suitable for running untrusted code. It maintains state between executions,
    allows for custom tools and functions to be made available to the code, and captures
    print outputs separately from return values.

    Args:
        additional_authorized_imports (`list[str]`):
            Additional authorized imports for the executor.
        max_print_outputs_length (`int`, defaults to `DEFAULT_MAX_LEN_OUTPUT=50_000`):
            Maximum length of the print outputs.
        additional_functions (`dict[str, callable]`, *optional*):
            Additional Python functions to be added to the executor.
    """
    def __init__(
        self,
        additional_authorized_imports: list[str],
        max_print_output_length: int | None = None,
        additional_functions: dict[str, callable] | None = None,
    ):
        self.custom_tools = {}
        self.state= {"__name__": "__main__"}
        self.max_print_output_length = max_print_output_length
        if max_print_output_length is None:
            self.max_print_output_length = DEFAULT_MAX_LEN_OUTPUT
        
        self.additional_authorized_imports = additional_authorized_imports
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        
        self._check_authorized_imports_are_installed()
        self.static_tools = None
        self.additional_functions = additional_functions or {}
        
    def _check_authorized_imports_are_installed(self):
        """
        Check that all authorized imports are installed on the system.

        Handles wildcard imports ("*") and partial star-pattern imports (e.g., "os.*").

        Raises:
            InterpreterError: If any of the authorized modules are not installed.
        """
        # find_spec returns None if the module is not installed
        missing_modules = [
            base_module for imp in self.authorized_imports if imp!="*" and find_spec(base_module := imp.split(".")[0]) is None
        ]
        if missing_modules:
            raise InterpreterError(
                f"Non-installed authorized modules: {', '.join(missing_modules)}. "
                f"Please install these modules or remove them from the authorized imports list."
            )
    
    def __call__(self, code_action: str) -> CodeOutput:
        output, is_final_answer = evaluate_python_code(
            code_action,
            static_tools=self.static_tools,
            custom_tools=self.custom_tools,
            state=self.state,
            authorized_imports=self.authorized_imports,
            max_print_output_length=self.max_print_output_length,
        )
        
        logs = str(self.state["_print_outputs"])
        
        return CodeOutput(output=output, logs=logs, is_final_answer=is_final_answer)
    
    def send_variables(self, variables: dict):
        self.state.update(variables)
        
    def send_tools(self, tools: dict[str, Callable]):
        """Send tools to the executor to make them available during code execution.
        
        Args:
            tools: Dictionary mapping tool names to callable functions.
        """
        # Combine agent tools, base Python tools, and additional Python functions
        self.static_tools = {**tools, **BASE_PYTHON_TOOLS.copy(), **self.additional_functions}
