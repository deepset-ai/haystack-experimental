# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from haystack.components.tools.tool_invoker import ToolInvoker
from haystack.core.serialization import default_from_dict, default_to_dict, import_class_by_name
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.agents.human_in_the_loop import (
    ConfirmationPolicy,
    ConfirmationStrategy,
    ConfirmationUI,
    ToolExecutionDecision,
)
from haystack_experimental.components.agents.human_in_the_loop.breakpoint import (
    _update_state_and_tool_call_messages_with_tool_execution_decisions,
)
from haystack_experimental.components.agents.human_in_the_loop.errors import ToolBreakpointException

if TYPE_CHECKING:
    from haystack_experimental.components.agents.agent import _ExecutionContext


class BlockingConfirmationStrategy:
    """
    Confirmation strategy that blocks execution to gather user feedback.
    """

    def __init__(self, confirmation_policy: ConfirmationPolicy, confirmation_ui: ConfirmationUI) -> None:
        """
        Initialize the BlockingConfirmationStrategy with a confirmation policy and UI.

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
        Serializes the BlockingConfirmationStrategy to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self, confirmation_policy=self.confirmation_policy.to_dict(), confirmation_ui=self.confirmation_ui.to_dict()
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlockingConfirmationStrategy":
        """
        Deserializes the BlockingConfirmationStrategy from a dictionary.

        :param data:
            Dictionary to deserialize from.

        :returns:
            Deserialized BlockingConfirmationStrategy.
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


class BreakpointConfirmationStrategy:
    """
    Confirmation strategy that raises a tool breakpoint exception to pause execution and gather user feedback.

    This strategy is designed for scenarios where immediate user interaction is not possible, such as in backend
    services. When a tool execution requires confirmation, it raises an `ToolBreakpointException`, which is caught
    by the Agent. The Agent can then serialize its current state, including the tool call details, and send this
    information to a front-end interface for user review.
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
            tool_id=tool_id,
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


def _handle_confirmation_strategies(
    *,
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> tuple[list[ChatMessage], "_ExecutionContext"]:
    """
    Prepare tool call parameters for execution and collect any error messages.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Messages containing tool calls to process
    :param execution_context: The current execution context containing state and inputs
    :returns: Tuple of modified messages with confirmed tool calls and tool call result messages
    """
    state = execution_context.state
    streaming_callback = execution_context.chat_generator_inputs.get("streaming_callback")
    tools_with_names = {tool.name: tool for tool in execution_context.tool_invoker_inputs["tools"]}
    enable_streaming_passthrough = execution_context.tool_invoker_inputs.get("enable_streaming_passthrough", False)
    existing_teds = execution_context.tool_execution_decisions if execution_context.tool_execution_decisions else []

    teds = []
    for message in messages_with_tool_calls:
        if not message.tool_calls:
            continue

        # confirmed_tool_calls = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.tool_name
            tool_to_invoke = tools_with_names[tool_name]

            # Combine user + state inputs
            final_args = ToolInvoker._inject_state_args(tool_to_invoke, tool_call.arguments.copy(), state)
            # Check whether to inject streaming_callback
            if (
                enable_streaming_passthrough
                and streaming_callback is not None
                and "streaming_callback" not in final_args
                and "streaming_callback" in ToolInvoker._get_func_params(tool_to_invoke)
            ):
                final_args["streaming_callback"] = streaming_callback

            # Get tool execution decisions from confirmation strategies
            # If no confirmation strategy is defined for this tool, proceed with execution
            if tool_name not in confirmation_strategies:
                teds.append(
                    ToolExecutionDecision(
                        tool_id=tool_call.id,
                        tool_name=tool_name,
                        execute=True,
                        final_tool_params=final_args,
                    )
                )
                continue

            # Check if there's already a decision for this tool call in the execution context
            ted = next((t for t in existing_teds if t.tool_id == tool_call.id), None)
            # If not, run the confirmation strategy
            if not ted:
                ted = confirmation_strategies[tool_name].run(
                    tool_name=tool_name, tool_description=tool_to_invoke.description, tool_params=final_args
                )
            teds.append(ted)

    new_state, modified_tool_call_messages = _update_state_and_tool_call_messages_with_tool_execution_decisions(
        state=state,
        tool_call_messages=messages_with_tool_calls,
        tool_execution_decisions=teds,
    )

    # Update execution context state
    execution_context.state = new_state

    return modified_tool_call_messages, execution_context
