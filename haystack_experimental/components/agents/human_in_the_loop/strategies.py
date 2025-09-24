# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_to_dict, import_class_by_name
from haystack.tools import Tool

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ToolExecutionDecision
from haystack_experimental.components.agents.human_in_the_loop.policies import ConfirmationPolicy
from haystack_experimental.components.agents.human_in_the_loop.user_interfaces import ConfirmationUI


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
