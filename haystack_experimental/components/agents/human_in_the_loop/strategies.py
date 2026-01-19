# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ToolExecutionDecision

from haystack_experimental.components.agents.human_in_the_loop import HITLBreakpointException


_REJECTION_FEEDBACK_TEMPLATE = "Tool execution for '{tool_name}' was rejected by the user."
_MODIFICATION_FEEDBACK_TEMPLATE = (
    "The parameters for tool '{tool_name}' were updated by the user to:\n{final_tool_params}"
)


class BreakpointConfirmationStrategy:
    """
    Confirmation strategy that raises a tool breakpoint exception to pause execution and gather user feedback.

    This strategy is designed for scenarios where immediate user interaction is not possible.
    When a tool execution requires confirmation, it raises an `HITLBreakpointException`, which is caught by the Agent.
    The Agent then serialize its current state, including the tool call details. This information can then be used to
    notify a user to review and confirm the tool execution.
    """

    def __init__(self, snapshot_file_path: str) -> None:
        """
        Initialize the BreakpointConfirmationStrategy.

        :param snapshot_file_path: The path to the directory that the snapshot should be saved.
        """
        self.snapshot_file_path = snapshot_file_path

    def run(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Run the breakpoint confirmation strategy for a given tool and its parameters.

        :param tool_name:
            The name of the tool to be executed.
        :param tool_description:
            The description of the tool.
        :param tool_params:
            The parameters to be passed to the tool.
        :param tool_call_id:
            Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
            specific tool invocation.
        :param confirmation_strategy_context:
            Optional dictionary for passing request-scoped resources. Not used by this strategy but included for
            interface compatibility.

        :raises HITLBreakpointException:
            Always raises an `HITLBreakpointException` exception to signal that user confirmation is required.

        :returns:
            This method does not return; it always raises an exception.
        """
        raise HITLBreakpointException(
            message=f"Tool execution for '{tool_name}' requires user confirmation.",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            snapshot_file_path=self.snapshot_file_path,
        )

    async def run_async(
        self,
        *,
        tool_name: str,
        tool_description: str,
        tool_params: dict[str, Any],
        tool_call_id: str | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
    ) -> ToolExecutionDecision:
        """
        Async version of run. Calls the sync run() method.

        :param tool_name:
            The name of the tool to be executed.
        :param tool_description:
            The description of the tool.
        :param tool_params:
            The parameters to be passed to the tool.
        :param tool_call_id:
            Optional unique identifier for the tool call.
        :param confirmation_strategy_context:
            Optional dictionary for passing request-scoped resources.

        :raises HITLBreakpointException:
            Always raises an `HITLBreakpointException` exception to signal that user confirmation is required.

        :returns:
            This method does not return; it always raises an exception.
        """
        return self.run(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_params=tool_params,
            tool_call_id=tool_call_id,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the BreakpointConfirmationStrategy to a dictionary.
        """
        return default_to_dict(self, snapshot_file_path=self.snapshot_file_path)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BreakpointConfirmationStrategy":
        """
        Deserializes the BreakpointConfirmationStrategy from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized BreakpointConfirmationStrategy.
        """
        return default_from_dict(cls, data)
