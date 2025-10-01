# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack.core.serialization import default_from_dict, default_to_dict, import_class_by_name

from haystack_experimental.components.agents.human_in_the_loop import (
    ConfirmationPolicy,
    ConfirmationUI,
    ToolExecutionDecision,
)
from haystack_experimental.components.agents.human_in_the_loop.errors import ToolBreakpointException


# TODO Consider renaming to BlockingConfirmationStrategy – emphasizes that execution waits for immediate confirmation.
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

    def run(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any], tool_id: Optional[str] = None
    ) -> ToolExecutionDecision:
        """
        Run the human-in-the-loop strategy for a given tool and its parameters.

        :param tool_name:
            The name of the tool to be executed.
        :param tool_description:
            The description of the tool.
        :param tool_params:
            The parameters to be passed to the tool.
        :param tool_id:
            Optional unique identifier for the tool.

        :returns:
            A ToolExecutionDecision indicating whether to execute the tool with the given parameters, or a
            feedback message if rejected.
        """
        # Check if we should ask based on policy
        if not self.confirmation_policy.should_ask(
            tool_name=tool_name, tool_description=tool_description, tool_params=tool_params
        ):
            return ToolExecutionDecision(
                tool_name=tool_name, execute=True, tool_id=tool_id, final_tool_params=tool_params
            )

        # Get user confirmation through UI
        confirmation_result = self.confirmation_ui.get_user_confirmation(tool_name, tool_description, tool_params)

        # Pass back the result to the policy for any learning/updating
        self.confirmation_policy.update_after_confirmation(
            tool_name, tool_description, tool_params, confirmation_result
        )

        # Process the confirmation result
        final_args = {}
        if confirmation_result.action == "reject":
            tool_result_message = f"Tool execution for '{tool_name}' rejected by user"
            if confirmation_result.feedback:
                tool_result_message += f" with feedback: {confirmation_result.feedback}"
            return ToolExecutionDecision(
                tool_name=tool_name, execute=False, tool_id=tool_id, feedback=tool_result_message
            )
        elif confirmation_result.action == "modify" and confirmation_result.new_tool_params:
            # Update the tool call params with the new params
            final_args.update(confirmation_result.new_tool_params)
            return ToolExecutionDecision(
                tool_name=tool_name,
                tool_id=tool_id,
                execute=True,
                feedback=f"The tool parameters for {tool_name} were modified by the user.",
                final_tool_params=final_args,
            )
        else:  # action == "confirm"
            return ToolExecutionDecision(
                tool_name=tool_name, execute=True, tool_id=tool_id, final_tool_params=tool_params
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the HumanInTheLoopStrategy to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self, confirmation_policy=self.confirmation_policy.to_dict(), confirmation_ui=self.confirmation_ui.to_dict()
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HumanInTheLoopStrategy":
        """
        Deserializes the HumanInTheLoopStrategy from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized HumanInTheLoopStrategy.
        """
        policy_data = data["init_parameters"]["confirmation_policy"]
        policy_class = import_class_by_name(policy_data["type"])
        if not hasattr(policy_class, "from_dict"):
            raise ValueError(f"Class {policy_class} does not implement from_dict method.")
        ui_data = data["init_parameters"]["confirmation_ui"]
        ui_class = import_class_by_name(ui_data["type"])
        if not hasattr(ui_class, "from_dict"):
            raise ValueError(f"Class {ui_class} does not implement from_dict method.")
        return cls(confirmation_policy=policy_class.from_dict(policy_data), confirmation_ui=ui_class.from_dict(ui_data))


# TODO Add a BreakpointConfirmationStrategy that:
#  1. Uses AgentBreakPoint to trigger a BreakpointException
#  2. Can resume from a snapshot and any modified tool parameters
#  - The user feedback gathering is pushed outside of this strategy b/c this is meant to be used in a backend service


class BreakpointConfirmationStrategy:
    """
    Confirmation strategy that raises a breakpoint exception to pause execution and gather user feedback.

    This strategy is designed for scenarios where immediate user interaction is not possible, such as in backend
    services. When a tool execution requires confirmation, it raises an `BreakpointException` exception, which can be
    caught by the surrounding system to handle user interaction separately.
    """

    def __init__(self, snapshot_file_path: str) -> None:
        """
        Initialize the BreakpointConfirmationStrategy.
        """
        self.snapshot_file_path = snapshot_file_path

    def run(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any], tool_id: Optional[str] = None
    ) -> ToolExecutionDecision:
        """
        Run the breakpoint confirmation strategy for a given tool and its parameters.

        :param tool_name:
            The name of the tool to be executed.
        :param tool_description:
            The description of the tool.
        :param tool_params:
            The parameters to be passed to the tool.
        :param tool_id:
            Optional unique identifier for the tool.

        :raises ToolBreakpointException:
            Always raises an `ToolBreakpointException` exception to signal that user confirmation is required.

        :returns:
            This method does not return; it always raises an exception.
        """
        raise ToolBreakpointException(
            message=f"Tool execution for '{tool_name}' requires user confirmation.",
            tool_name=tool_name,
            snapshot_file_path=self.snapshot_file_path,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialization is not implemented for BreakpointConfirmationStrategy.

        :raises NotImplementedError:
            Always raises this exception since serialization is not supported.
        """
        raise default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BreakpointConfirmationStrategy":
        """
        Deserialization is not implemented for BreakpointConfirmationStrategy.

        :param data:
            Dictionary to deserialize from.

        :raises NotImplementedError:
            Always raises this exception since deserialization is not supported.
        """
        raise default_from_dict(cls, data)
