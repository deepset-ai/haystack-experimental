# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from threading import Lock
from typing import Any, Optional

from haystack.core.serialization import default_to_dict
from haystack.tools import Tool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ConfirmationUIResult
from haystack_experimental.components.agents.human_in_the_loop.types import ConfirmationUI

_ui_lock = Lock()


class RichConsoleUI(ConfirmationUI):
    """Rich console interface for user interaction."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def get_user_confirmation(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Get user confirmation for tool execution via rich console prompts.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        :returns: ConfirmationUIResult based on user input.
        """
        with _ui_lock:
            self._display_tool_info(tool, tool_params)
            # y: confirm, n: reject, m: modify
            choice = Prompt.ask("\nYour choice", choices=["y", "n", "m"], default="y")
            return self._process_choice(choice, tool_params)

    def _display_tool_info(self, tool: Tool, tool_params: dict[str, Any]) -> None:
        """
        Display tool information and parameters in a rich panel.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        """
        lines = [f"[bold yellow]Tool:[/bold yellow] {tool.name}"]

        if hasattr(tool, "description") and tool.description:
            lines.append(f"[bold yellow]Description:[/bold yellow] {tool.description}")

        if tool_params:
            lines.append("\n[bold yellow]Arguments:[/bold yellow]")
            for k, v in tool_params.items():
                lines.append(f"[cyan]{k}:[/cyan] {v}")

        self.console.print(Panel("\n".join(lines), title="🔧 Tool Execution Request", title_align="left"))

    def _process_choice(self, choice: str, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Process the user's choice and return the corresponding ConfirmationUIResult.

        :param choice: The user's choice ('y', 'n', or 'm').
        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult based on user input.
        """
        if choice == "y":
            return ConfirmationUIResult(action="confirm")
        elif choice == "m":
            return self._modify_params(tool_params)
        else:  # reject
            feedback = Prompt.ask("Feedback message (optional)", default="")
            return ConfirmationUIResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Prompt the user to modify tool parameters.

        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult with modified parameters.
        """
        new_params: dict[str, Any] = {}
        for k, v in tool_params.items():
            new_val = Prompt.ask(f"Modify '{k}'", default=json.dumps(v))
            try:
                # Try to parse JSON back into original type
                parsed = json.loads(new_val)
                new_params[k] = parsed
            except (json.JSONDecodeError, TypeError):
                # Fallback to raw string if not valid JSON
                new_params[k] = new_val

        return ConfirmationUIResult(action="modify", new_tool_params=new_params)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the RichConsoleConfirmationUI to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        # Note: Console object is not serializable; we store None
        return default_to_dict(self, console=None)


class SimpleConsoleUI(ConfirmationUI):
    """Simple console interface using standard input/output."""

    def get_user_confirmation(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Get user confirmation for tool execution via simple console prompts.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        """
        with _ui_lock:
            self._display_tool_info(tool, tool_params)
            choice = input("Confirm execution? (y=confirm / n=reject / m=modify): ").strip().lower()
            return self._process_choice(choice, tool_params)

    def _display_tool_info(self, tool: Tool, tool_params: dict[str, Any]) -> None:
        """
        Display tool information and parameters in the console.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        """
        print("\n--- Tool Execution Request ---")
        print(f"Tool: {tool.name}")

        if hasattr(tool, "description") and tool.description:
            print(f"Description: {tool.description}")

        if tool_params:
            print("Arguments:")
            for k, v in tool_params.items():
                print(f"  {k}: {v}")
        print("-" * 30)

    def _process_choice(self, choice: str, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Process the user's choice and return the corresponding ConfirmationUIResult.

        :param choice: The user's choice ('y', 'n', or 'm').
        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult based on user input.
        """
        if choice in ("y", "yes"):
            return ConfirmationUIResult(action="confirm")
        elif choice in ("m", "modify"):
            return self._modify_params(tool_params)
        else:  # reject
            feedback = input("Feedback message (optional): ").strip()
            return ConfirmationUIResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Prompt the user to modify tool parameters.

        :param tool_params: The original tool parameters.
        :returns:
            ConfirmationUIResult with modified parameters.
        """
        new_params: dict[str, Any] = {}
        for k, v in tool_params.items():
            new_val = Prompt.ask(f"Modify '{k}'", default=json.dumps(v))
            try:
                # Try to parse JSON back into original type
                parsed = json.loads(new_val)
                new_params[k] = parsed
            except (json.JSONDecodeError, TypeError):
                # Fallback to raw string if not valid JSON
                new_params[k] = new_val

        return ConfirmationUIResult(action="modify", new_tool_params=new_params)
