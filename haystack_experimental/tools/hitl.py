# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

from haystack.core.serialization import default_from_dict, default_to_dict, import_class_by_name
from haystack.tools import Tool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


@dataclass
class ConfirmationUIResult:
    """
    Result of the confirmation UI interaction.

    :param action:
        The action taken by the user such as "confirm", "reject", or "modify".
        This action type is not enforced to allow for custom actions to be implemented.
    :param feedback:
        Optional feedback message from the user. For example, if the user rejects the tool execution,
        they might provide a reason for the rejection.
    :param new_tool_params:
        Optional set of new parameters for the tool. For example, if the user chooses to modify the tool parameters,
        they can provide a new set of parameters here.
    """

    action: str  # "confirm", "reject", "modify"
    feedback: Optional[str] = None
    new_tool_params: Optional[dict[str, Any]] = None


@dataclass
class ToolExecutionDecision:
    """
    Decision made regarding tool execution.

    :param tool_name:
        The name of the tool to be executed.
    :param feedback:
        Optional feedback message if the tool execution was rejected.
    :param final_tool_params:
        Optional final parameters for the tool if execution is confirmed or modified.
    """

    tool_name: str
    feedback: Optional[str] = None
    final_tool_params: Optional[dict[str, Any]] = None


# Confirmation policy implementations
class ConfirmationPolicy:
    """Base class for confirmation policies."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """Determine whether to ask for confirmation."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy":
        """Deserialize the policy from a dictionary."""
        return default_from_dict(cls, data)


class AlwaysAskPolicy(ConfirmationPolicy):
    """Always ask for confirmation."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """
        Always ask for confirmation before executing the tool.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns True, indicating confirmation is needed.
        """
        return True


class NeverAskPolicy(ConfirmationPolicy):
    """Never ask for confirmation."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """
        Never ask for confirmation, always proceed with tool execution.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns False, indicating no confirmation is needed.
        """
        return False


class AskOncePolicy(ConfirmationPolicy):
    """Ask only once per tool with specific parameters."""

    def __init__(self):
        self._asked_tools = {}

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """
        Ask for confirmation only once per tool with specific parameters.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        :returns: True if confirmation is needed, False if already asked with the same parameters.
        """
        if tool.name in self._asked_tools and self._asked_tools[tool.name] == tool_params:
            return False
        self._asked_tools[tool.name] = tool_params
        return True


# Confirmation UI implementations
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


class RichConsoleConfirmationUI(ConfirmationUI):
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
        self._display_tool_info(tool, tool_params)

        choice = Prompt.ask(
            "\nYour choice",
            choices=["y", "n", "m"],  # confirm, reject, modify
            default="y",
        )

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


class SimpleConsoleConfirmationUI(ConfirmationUI):
    """Simple console interface using standard input/output."""

    def get_user_confirmation(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """
        Get user confirmation for tool execution via simple console prompts.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        """
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


# Human-in-the-loop strategy
class HumanInTheLoopStrategy:
    """
    Human-in-the-loop strategy for tool execution confirmation.
    """

    def __init__(self, confirmation_policy: ConfirmationPolicy, confirmation_ui: ConfirmationUI) -> None:
        """
        Initialize the HumanInTheLoopStrategy with a confirmation policy and UI.

        :param confirmation_policy:
            The confirmation policy to determine when to ask for user confirmation.
        :param confirmation_ui:
            The user interface to interact with the user for confirmation.
        """
        self.confirmation_policy = confirmation_policy
        self.confirmation_ui = confirmation_ui

    def run(self, tool: Tool, tool_params: dict[str, Any]) -> ToolExecutionDecision:
        """
        Run the human-in-the-loop strategy for a given tool and its parameters.

        :param tool:
            The tool to be confirmed.
        :param tool_params:
            The parameters to be passed to the tool.

        :returns:
            A ToolExecutionDecision indicating whether to execute the tool with the given parameters, or a
            feedback message if rejected.
        """
        # Check if we should ask based on policy
        if not self.confirmation_policy.should_ask(tool, tool_params):
            return ToolExecutionDecision(tool_name=tool.name, final_tool_params=tool_params)

        # Get user confirmation through UI
        confirmation_result = self.confirmation_ui.get_user_confirmation(tool, tool_params)

        # Process the confirmation result
        final_args = {}
        if confirmation_result.action == "reject":
            tool_result_message = f"Tool execution for '{tool.name}' rejected by user"
            if confirmation_result.feedback:
                tool_result_message += f" with feedback: {confirmation_result.feedback}"
            return ToolExecutionDecision(tool_name=tool.name, feedback=tool_result_message)
        elif confirmation_result.action == "modify" and confirmation_result.new_tool_params:
            # Update the tool call params with the new params
            final_args.update(confirmation_result.new_tool_params)
            return ToolExecutionDecision(tool_name=tool.name, final_tool_params=final_args)
        else:  # action == "confirm"
            return ToolExecutionDecision(tool_name=tool.name, final_tool_params=tool_params)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the HumanInTheLoopStrategy to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, policy=self.confirmation_policy.to_dict(), ui=self.confirmation_ui.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanInTheLoopStrategy":
        """
        Deserializes the HumanInTheLoopStrategy from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized HumanInTheLoopStrategy.
        """
        policy_data = data["data"]["policy"]
        ui_data = data["data"]["ui"]

        policy_class = import_class_by_name(policy_data["type"])
        if not issubclass(policy_class, ConfirmationPolicy):
            raise TypeError(f"Class '{policy_class}' is not a subclass of ConfirmationPolicy")

        ui_class = import_class_by_name(ui_data["type"])
        if not issubclass(ui_class, ConfirmationUI):
            raise TypeError(f"Class '{ui_class}' is not a subclass of ConfirmationUI")

        return cls(confirmation_policy=policy_class.from_dict(policy_data), confirmation_ui=ui_class.from_dict(ui_data))
