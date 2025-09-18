# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

from haystack.tools import Tool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from haystack.core.serialization import generate_qualified_class_name, import_class_by_name

from haystack_experimental.tools.types.protocol import ConfirmationPolicy, UserInterface


@dataclass
class ConfirmationResult:
    """
    Result of the confirmation process.

    :param action:
        The action taken by the user: "confirm", "reject", or "modify".
    :param feedback:
        If the action is "reject", an optional feedback message from the user.
    :param new_tool_params:
        If the action is "modify", the new parameters for the tool.
    """

    action: str  # "confirm", "reject", "modify"
    feedback: Optional[str] = None
    new_tool_params: Optional[dict[str, Any]] = None


# Confirmation policy implementations
class AlwaysAskPolicy:
    """Always ask for confirmation."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"type": generate_qualified_class_name(type(self)), "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlwaysAskPolicy":
        return cls()


class NeverAskPolicy:
    """Never ask for confirmation."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"type": generate_qualified_class_name(type(self)), "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeverAskPolicy":
        return cls()


class AskOncePolicy:
    """Ask only once per tool type."""

    def __init__(self):
        self._asked_tools = set()

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        if tool.name in self._asked_tools:
            return False
        self._asked_tools.add(tool.name)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {"type": generate_qualified_class_name(type(self)), "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AskOncePolicy":
        return cls()


class RichConsoleUI:
    """Rich console interface for user interaction."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def get_user_confirmation(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationResult:
        self._display_tool_info(tool, tool_params)

        choice = Prompt.ask(
            "\nYour choice",
            choices=["y", "n", "m"],  # confirm, reject, modify
            default="y",
        )

        return self._process_choice(choice, tool_params)

    def _display_tool_info(self, tool: Tool, tool_params: dict[str, Any]) -> None:
        lines = [f"[bold yellow]Tool:[/bold yellow] {tool.name}"]

        if hasattr(tool, 'description') and tool.description:
            lines.append(f"[bold yellow]Description:[/bold yellow] {tool.description}")

        if tool_params:
            lines.append("\n[bold yellow]Arguments:[/bold yellow]")
            for k, v in tool_params.items():
                lines.append(f"[cyan]{k}:[/cyan] {v}")

        self.console.print(Panel(
            "\n".join(lines),
            title="🔧 Tool Execution Request",
            title_align="left"
        ))

    def _process_choice(self, choice: str, tool_params: dict[str, Any]) -> ConfirmationResult:
        if choice == "y":
            return ConfirmationResult(action="confirm")
        elif choice == "m":
            return self._modify_params(tool_params)
        else:  # reject
            feedback = Prompt.ask("Feedback message (optional)", default="")
            return ConfirmationResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationResult:
        new_params = {}
        for k, v in tool_params.items():
            new_val = Prompt.ask(f"Modify '{k}'", default=str(v))
            # Try to preserve original type
            try:
                if isinstance(v, bool):
                    new_params[k] = new_val.lower() in ('true', 'yes', '1', 'on')
                elif isinstance(v, int):
                    new_params[k] = int(new_val)
                elif isinstance(v, float):
                    new_params[k] = float(new_val)
                else:
                    new_params[k] = new_val
            except (ValueError, TypeError):
                new_params[k] = new_val

        return ConfirmationResult(action="modify", new_tool_params=new_params)

    def to_dict(self) -> dict[str, Any]:
        # Note: Console object is not serializable; we store None
        return {"type": generate_qualified_class_name(type(self)), "data": {"console": None}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RichConsoleUI":
        return cls(console=None)


class SimpleConsoleUI:
    """Simple console interface using standard input/output."""

    def get_user_confirmation(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationResult:
        self._display_tool_info(tool, tool_params)

        choice = input("Confirm execution? (y=confirm / n=reject / m=modify): ").strip().lower()

        return self._process_choice(choice, tool_params)

    def _display_tool_info(self, tool: Tool, tool_params: dict[str, Any]) -> None:
        print(f"\n--- Tool Execution Request ---")
        print(f"Tool: {tool.name}")

        if hasattr(tool, 'description') and tool.description:
            print(f"Description: {tool.description}")

        if tool_params:
            print("Arguments:")
            for k, v in tool_params.items():
                print(f"  {k}: {v}")
        print("-" * 30)

    def _process_choice(self, choice: str, tool_params: dict[str, Any]) -> ConfirmationResult:
        if choice in ("y", "yes"):
            return ConfirmationResult(action="confirm")
        elif choice in ("m", "modify"):
            return self._modify_params(tool_params)
        else:  # reject
            feedback = input("Feedback message (optional): ").strip()
            return ConfirmationResult(action="reject", feedback=feedback or None)

    def _modify_params(self, tool_params: dict[str, Any]) -> ConfirmationResult:
        new_params = {}
        for k, v in tool_params.items():
            new_val = input(f"Modify '{k}' [{v}]: ").strip() or v
            # Try to preserve original type
            try:
                if isinstance(v, bool):
                    new_params[k] = new_val.lower() in ('true', 'yes', '1', 'on')
                elif isinstance(v, int):
                    new_params[k] = int(new_val)
                elif isinstance(v, float):
                    new_params[k] = float(new_val)
                else:
                    new_params[k] = new_val
            except (ValueError, TypeError):
                new_params[k] = new_val

        return ConfirmationResult(action="modify", new_tool_params=new_params)

    def to_dict(self) -> dict[str, Any]:
        return {"type": generate_qualified_class_name(type(self)), "data": {}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimpleConsoleUI":
        return cls()


class HumanInTheLoopStrategy:
    """
    Main strategy that orchestrates the confirmation process.
    Separates policy decisions from UI interactions.
    """

    def __init__(self, policy: ConfirmationPolicy, ui: UserInterface):
        self.policy = policy
        self.ui = ui

    def run(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationResult:
        """
        Main entry point for human-in-the-loop confirmation.
        """
        # Check if we should ask based on policy
        if not self.policy.should_ask(tool, tool_params):
            return ConfirmationResult(action="confirm")

        # Get user confirmation through UI
        return self.ui.get_user_confirmation(tool, tool_params)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "policy": self.policy.to_dict(),
                "ui": self.ui.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanInTheLoopStrategy":
        policy_data = data["data"]["policy"]
        ui_data = data["data"]["ui"]

        policy_class = import_class_by_name(policy_data["type"])
        if not hasattr(policy_class, "from_dict") or not callable(getattr(policy_class, "from_dict")):
            raise TypeError(f"Class '{policy_class}' does not have a 'from_dict' method")

        ui_class = import_class_by_name(ui_data["type"])
        if not hasattr(ui_class, "from_dict") or not callable(getattr(ui_class, "from_dict")):
            raise TypeError(f"Class '{ui_class}' does not have a 'from_dict' method")

        return cls(policy=policy_class.from_dict(policy_data), ui=ui_class.from_dict(ui_data))
