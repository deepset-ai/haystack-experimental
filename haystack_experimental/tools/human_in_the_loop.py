# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any, Optional

from haystack.tools import Tool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from haystack_experimental.tools.types.protocol import ConfirmationPrompt, ExecutionPolicy


@dataclass
class ConfirmationResult:
    """
    Result of the confirmation prompt to capture a user's decision.

    :param action: The action chosen by the user (e.g. "confirm", "reject", or "modify").
    :param feedback: Optional feedback message if the action is "reject".
    :param new_params: Optional new parameters if the action is "modify".
    """

    action: str  # This is left as a string to allow users to define their own actions if needed.
    feedback: Optional[str] = None
    new_params: Optional[dict[str, Any]] = None


class RichConsolePrompt:
    """
    Confirmation prompt using Rich library for enhanced console interaction.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """
        :param console: Optional Rich Console instance. If None, a new Console will be created.
        """
        self.console = console or Console()

    def confirm(self, tool_name: str, params: dict[str, Any]) -> ConfirmationResult:
        """
        Ask for user confirmation before executing a tool.

        :param tool_name: Name of the tool to be executed.
        :param params: Parameters to be passed to the tool.
        :returns:
            ConfirmationResult with action (e.g. "confirm" or "reject"), optional feedback message and new parameters
            if modified.
        """
        # Display info
        lines = [f"[bold yellow]Tool:[/bold yellow] {tool_name}"]
        if params:
            lines.append("\n[bold yellow]Arguments:[/bold yellow]")
            for k, v in params.items():
                lines.append(f"\n[cyan]{k}:[/cyan]\n  {v}")
        self.console.print(Panel("\n".join(lines), title="🔧 Tool Execution Request"))

        # Ask action
        choice = Prompt.ask(
            "\nYour choice",
            choices=["y", "n", "m"],  # confirm, reject, modify
            default="y",
        )
        if choice == "y":
            return ConfirmationResult(action="confirm")
        elif choice == "m":
            new_params = {}
            for k, v in params.items():
                new_val = Prompt.ask(f"Modify '{k}'", default=str(v))
                new_params[k] = new_val
            return ConfirmationResult(action="modify", new_params=new_params)
        else:  # reject
            feedback = Prompt.ask("Feedback message (optional)", default="")
            return ConfirmationResult(action="reject", feedback=feedback or None)


class SimpleInputPrompt:
    """
    Simple confirmation prompt using standard input/output.
    """

    def confirm(self, tool_name: str, params: dict[str, Any]) -> ConfirmationResult:
        """
        Ask for user confirmation before executing a tool.

        :param tool_name: Name of the tool to be executed.
        :param params: Parameters to be passed to the tool.
        :returns:
            ConfirmationResult with action (e.g. "confirm" or "reject"), optional feedback message and new parameters
            if modified.
        """
        print(f"Tool: {tool_name}")
        if params:
            print("Arguments:")
            for k, v in params.items():
                print(f"  {k}: {v}")

        choice = input("Confirm execution? (y=confirm / n=reject / m=modify): ").strip().lower()
        if choice == "y":
            return ConfirmationResult(action="confirm")
        elif choice == "m":
            new_params = {}
            for k, v in params.items():
                new_val = input(f"Modify '{k}' [{v}]: ").strip() or v
                new_params[k] = new_val
            return ConfirmationResult(action="modify", new_params=new_params)
        else:  # modify
            feedback = input("Feedback message (optional): ").strip()
            return ConfirmationResult(action="reject", feedback=feedback or None)


class DefaultPolicy:
    """
    Default execution policy:

    - If confirmed, run the tool with original params.
    - If rejected, return a rejection message.
    - If modified, run the tool immediately with new params.
    """

    def handle(self, result: ConfirmationResult, tool: Tool, kwargs: dict[str, Any]) -> Any:
        """
        Handle the confirmation result and execute the tool accordingly.

        :param result: The result from the confirmation prompt.
        :param tool: The tool to potentially execute.
        :param kwargs: The original parameters for the tool.

        :returns:
            The result of the tool execution or a rejection message.
        """
        if result.action == "reject":
            return {
                "status": "rejected",
                "tool": tool.name,
                "feedback": result.feedback or "Tool execution rejected by user",
            }
        elif result.action == "modify" and result.new_params:
            # Run immediately with new params
            return tool.function(**result.new_params)
        return tool.function(**kwargs)


class AutoConfirmPolicy:
    """
    Always confirm and run the tool, ignoring user input.
    """

    def handle(self, result: ConfirmationResult, tool: Tool, kwargs: dict[str, Any]) -> Any:
        """
        Always execute the tool, ignoring any rejection from the user.

        :param result: The result from the confirmation prompt (ignored).
        :param tool: The tool to execute.
        :param kwargs: The original parameters for the tool.

        :returns: The result of the tool execution.
        """
        # Always run, ignore user rejection
        return tool.function(**kwargs)


def confirmation_wrapper(
    tool: Tool,
    strategy: ConfirmationPrompt,
    policy: ExecutionPolicy = DefaultPolicy(),
) -> Tool:
    """
    Wrap a tool with a human-in-the-loop confirmation step.

    :param tool: The tool to wrap.
    :param strategy: The confirmation prompt strategy to use.
    :param policy: The execution policy to apply based on user input.
    :return: A new Tool instance with confirmation logic.
    """

    def wrapped_function(**kwargs: Any) -> Any:
        result = strategy.confirm(tool.name, kwargs)
        return policy.handle(result, tool, kwargs)

    return replace(tool, function=wrapped_function)
