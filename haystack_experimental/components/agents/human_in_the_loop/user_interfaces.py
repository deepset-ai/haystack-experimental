# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from threading import Lock
from typing import Any, Optional

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.tools import Tool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ConfirmationUIResult

_ui_lock = Lock()


class ConfirmationUI:
    """Base class for confirmation UIs."""

    def get_user_confirmation(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """Get user confirmation for tool execution."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize the UI to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationUI":
        """Deserialize the ConfirmationUI from a dictionary."""
        return default_from_dict(cls, data)


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
        lines = [f"[bold yellow]Tool:[/bold yellow] {tool.name}"]

        if hasattr(tool, "description") and tool.description:
            lines.append(f"[bold yellow]Description:[/bold yellow] {tool.description}")

        if tool_params:
            lines.append("\n[bold yellow]Arguments:[/bold yellow]")
            for k, v in tool_params.items():
                lines.append(f"[cyan]{k}:[/cyan] {v}")

        self.console.print(Panel("\n".join(lines), title="🔧 Tool Execution Request", title_align="left"))

    def _process_choice(self, choice: str, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        if choice == "y":
            return ConfirmationUIResult(action="confirm")
        elif choice == "m":
            return self._modify_params(tool_params)
        else:  # reject
            feedback = Prompt.ask("Feedback message (optional)", default="")
            return ConfirmationUIResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        new_params: dict[str, Any] = {}
        for k, v in tool_params.items():
            new_val = Prompt.ask(f"Modify '{k}'", default=str(v))
            # Try to preserve original type
            try:
                if isinstance(v, bool):
                    new_params[k] = new_val.lower() in ("true", "yes", "1", "on")
                elif isinstance(v, int):
                    new_params[k] = int(new_val)
                elif isinstance(v, float):
                    new_params[k] = float(new_val)
                else:
                    new_params[k] = new_val
            except (ValueError, TypeError):
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
        if choice in ("y", "yes"):
            return ConfirmationUIResult(action="confirm")
        elif choice in ("m", "modify"):
            return self._modify_params(tool_params)
        else:  # reject
            feedback = input("Feedback message (optional): ").strip()
            return ConfirmationUIResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        new_params: dict[str, Any] = {}
        for k, v in tool_params.items():
            new_val = input(f"Modify '{k}' [{v}]: ").strip() or v
            # Try to preserve original type
            try:
                if isinstance(v, bool):
                    new_params[k] = new_val.lower() in ("true", "yes", "1", "on")
                elif isinstance(v, int):
                    new_params[k] = int(new_val)
                elif isinstance(v, float):
                    new_params[k] = float(new_val)
                else:
                    new_params[k] = new_val
            except (ValueError, TypeError):
                new_params[k] = new_val

        return ConfirmationUIResult(action="modify", new_tool_params=new_params)
