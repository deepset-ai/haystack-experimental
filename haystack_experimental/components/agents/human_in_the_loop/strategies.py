# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional

from haystack.components.agents.state import State
from haystack.components.tools.tool_invoker import ToolInvoker
from haystack.core.serialization import default_from_dict, default_to_dict, import_class_by_name
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool

from haystack_experimental.components.agents.human_in_the_loop import (
    ConfirmationPolicy,
    ConfirmationStrategy,
    ConfirmationUI,
    HITLBreakpointException,
    ToolExecutionDecision,
)

if TYPE_CHECKING:
    from haystack_experimental.components.agents.agent import _ExecutionContext


_REJECTION_FEEDBACK_TEMPLATE = "Tool execution for '{tool_name}' was rejected by the user."
_MODIFICATION_FEEDBACK_TEMPLATE = (
    "The parameters for tool '{tool_name}' were updated by the user to:\n{final_tool_params}"
)


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
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any], tool_call_id: Optional[str] = None
    ) -> ToolExecutionDecision:
        """
        Run the human-in-the-loop strategy for a given tool and its parameters.

        :param tool_name:
            The name of the tool to be executed.
        :param tool_description:
            The description of the tool.
        :param tool_params:
            The parameters to be passed to the tool.
        :param tool_call_id:
            Optional unique identifier for the tool call. This can be used to track and correlate the decision with a
            specific tool invocation.

        :returns:
            A ToolExecutionDecision indicating whether to execute the tool with the given parameters, or a
            feedback message if rejected.
        """
        # Check if we should ask based on policy
        if not self.confirmation_policy.should_ask(
            tool_name=tool_name, tool_description=tool_description, tool_params=tool_params
        ):
            return ToolExecutionDecision(
                tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
            )

        # Get user confirmation through UI
        confirmation_ui_result = self.confirmation_ui.get_user_confirmation(tool_name, tool_description, tool_params)

        # Pass back the result to the policy for any learning/updating
        self.confirmation_policy.update_after_confirmation(
            tool_name, tool_description, tool_params, confirmation_ui_result
        )

        # Process the confirmation result
        final_args = {}
        if confirmation_ui_result.action == "reject":
            explanation_text = _REJECTION_FEEDBACK_TEMPLATE.format(tool_name=tool_name)
            if confirmation_ui_result.feedback:
                explanation_text += f" With feedback: {confirmation_ui_result.feedback}"
            return ToolExecutionDecision(
                tool_name=tool_name, execute=False, tool_call_id=tool_call_id, feedback=explanation_text
            )
        elif confirmation_ui_result.action == "modify" and confirmation_ui_result.new_tool_params:
            # Update the tool call params with the new params
            final_args.update(confirmation_ui_result.new_tool_params)
            explanation_text = _MODIFICATION_FEEDBACK_TEMPLATE.format(tool_name=tool_name, final_tool_params=final_args)
            if confirmation_ui_result.feedback:
                explanation_text += f" With feedback: {confirmation_ui_result.feedback}"
            return ToolExecutionDecision(
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                execute=True,
                feedback=explanation_text,
                final_tool_params=final_args,
            )
        else:  # action == "confirm"
            return ToolExecutionDecision(
                tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
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
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any], tool_call_id: Optional[str] = None
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


def _prepare_tool_args(
    *,
    tool: Tool,
    tool_call_arguments: dict[str, Any],
    state: State,
    streaming_callback: Optional[StreamingCallbackT] = None,
    enable_streaming_passthrough: bool = False,
) -> dict[str, Any]:
    """
    Prepare the final arguments for a tool by injecting state inputs and optionally a streaming callback.

    :param tool:
        The tool instance to prepare arguments for.
    :param tool_call_arguments:
        The initial arguments provided for the tool call.
    :param state:
        The current state containing inputs to be injected into the tool arguments.
    :param streaming_callback:
        Optional streaming callback to be injected if enabled and applicable.
    :param enable_streaming_passthrough:
        Flag indicating whether to inject the streaming callback into the tool arguments.

    :returns:
        A dictionary of final arguments ready for tool invocation.
    """
    # Combine user + state inputs
    final_args = ToolInvoker._inject_state_args(tool, tool_call_arguments.copy(), state)
    # Check whether to inject streaming_callback
    if (
        enable_streaming_passthrough
        and streaming_callback is not None
        and "streaming_callback" not in final_args
        and "streaming_callback" in ToolInvoker._get_func_params(tool)
    ):
        final_args["streaming_callback"] = streaming_callback
    return final_args


def _process_confirmation_strategies(
    *,
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """
    Run the confirmation strategies and return modified tool call messages and updated chat history.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Chat messages containing tool calls
    :param execution_context: The current execution context of the agent
    :returns:
        Tuple of modified messages with confirmed tool calls and updated chat history
    """
    # Run confirmation strategies and get tool execution decisions
    teds = _run_confirmation_strategies(
        confirmation_strategies=confirmation_strategies,
        messages_with_tool_calls=messages_with_tool_calls,
        execution_context=execution_context,
    )

    # Apply tool execution decisions to messages_with_tool_calls
    rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
        tool_call_messages=messages_with_tool_calls,
        tool_execution_decisions=teds,
    )

    # Update the chat history with rejection messages and new tool call messages
    new_chat_history = _update_chat_history(
        chat_history=execution_context.state.get("messages"),
        rejection_messages=rejection_messages,
        tool_call_and_explanation_messages=modified_tool_call_messages,
    )

    return modified_tool_call_messages, new_chat_history


def _run_confirmation_strategies(
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> list[ToolExecutionDecision]:
    """
    Run confirmation strategies for tool calls in the provided chat messages.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Messages containing tool calls to process
    :param execution_context: The current execution context containing state and inputs
    :returns:
        A list of ToolExecutionDecision objects representing the decisions made for each tool call.
    """
    state = execution_context.state
    tools_with_names = {tool.name: tool for tool in execution_context.tool_invoker_inputs["tools"]}
    existing_teds = execution_context.tool_execution_decisions if execution_context.tool_execution_decisions else []
    existing_teds_by_name = {ted.tool_name: ted for ted in existing_teds if ted.tool_name}
    existing_teds_by_id = {ted.tool_call_id: ted for ted in existing_teds if ted.tool_call_id}

    teds = []
    for message in messages_with_tool_calls:
        if not message.tool_calls:
            continue

        for tool_call in message.tool_calls:
            tool_name = tool_call.tool_name
            tool_to_invoke = tools_with_names[tool_name]

            # Prepare final tool args
            final_args = _prepare_tool_args(
                tool=tool_to_invoke,
                tool_call_arguments=tool_call.arguments,
                state=state,
                streaming_callback=execution_context.tool_invoker_inputs.get("streaming_callback"),
                enable_streaming_passthrough=execution_context.tool_invoker_inputs.get(
                    "enable_streaming_passthrough", False
                ),
            )

            # Get tool execution decisions from confirmation strategies
            # If no confirmation strategy is defined for this tool, proceed with execution
            if tool_name not in confirmation_strategies:
                teds.append(
                    ToolExecutionDecision(
                        tool_call_id=tool_call.id,
                        tool_name=tool_name,
                        execute=True,
                        final_tool_params=final_args,
                    )
                )
                continue

            # Check if there's already a decision for this tool call in the execution context
            ted = existing_teds_by_id.get(tool_call.id or "") or existing_teds_by_name.get(tool_name)

            # If not, run the confirmation strategy
            if not ted:
                ted = confirmation_strategies[tool_name].run(
                    tool_name=tool_name, tool_description=tool_to_invoke.description, tool_params=final_args
                )
            teds.append(ted)

    return teds


def _apply_tool_execution_decisions(
    tool_call_messages: list[ChatMessage], tool_execution_decisions: list[ToolExecutionDecision]
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """
    Apply the tool execution decisions to the tool call messages.

    :param tool_call_messages: The tool call messages to apply the decisions to.
    :param tool_execution_decisions: The tool execution decisions to apply.
    :returns:
        A tuple containing:
        - A list of rejection messages for rejected tool calls. These are pairs of tool call and tool call result
          messages.
        - A list of tool call messages for confirmed or modified tool calls. If tool parameters were modified,
          a user message explaining the modification is included before the tool call message.
    """
    decision_by_id = {d.tool_call_id: d for d in tool_execution_decisions if d.tool_call_id}
    decision_by_name = {d.tool_name: d for d in tool_execution_decisions if d.tool_name}

    def make_assistant_message(chat_message, tool_calls):
        return ChatMessage.from_assistant(
            text=chat_message.text,
            meta=chat_message.meta,
            name=chat_message.name,
            tool_calls=tool_calls,
            reasoning=chat_message.reasoning,
        )

    new_tool_call_messages = []
    rejection_messages = []

    for chat_msg in tool_call_messages:
        new_tool_calls = []
        for tc in chat_msg.tool_calls or []:
            ted = decision_by_id.get(tc.id or "") or decision_by_name.get(tc.tool_name)
            if not ted:
                # This shouldn't happen, if so something went wrong in _run_confirmation_strategies
                continue

            if not ted.execute:
                # rejected tool call
                tool_result_text = ted.feedback or _REJECTION_FEEDBACK_TEMPLATE.format(tool_name=tc.tool_name)
                rejection_messages.extend(
                    [
                        make_assistant_message(chat_msg, [tc]),
                        ChatMessage.from_tool(tool_result=tool_result_text, origin=tc, error=True),
                    ]
                )
                continue

            # Covers confirm and modify cases
            final_args = ted.final_tool_params or {}
            if tc.arguments != final_args:
                # In the modify case we add a user message explaining the modification otherwise the LLM won't know
                # why the tool parameters changed and will likely just try and call the tool again with the
                # original parameters.
                user_text = ted.feedback or _MODIFICATION_FEEDBACK_TEMPLATE.format(
                    tool_name=tc.tool_name, final_tool_params=final_args
                )
                new_tool_call_messages.append(ChatMessage.from_user(text=user_text))
            new_tool_calls.append(replace(tc, arguments=final_args))

        # Only add the tool call message if there are any tool calls left (i.e. not all were rejected)
        if new_tool_calls:
            new_tool_call_messages.append(make_assistant_message(chat_msg, new_tool_calls))

    return rejection_messages, new_tool_call_messages


def _update_chat_history(
    chat_history: list[ChatMessage],
    rejection_messages: list[ChatMessage],
    tool_call_and_explanation_messages: list[ChatMessage],
) -> list[ChatMessage]:
    """
    Update the chat history to include rejection messages and tool call messages at the appropriate positions.

    Steps:
    1. Identify the last user message and the last tool message in the current chat history.
    2. Determine the insertion point as the maximum index of these two messages.
    3. Create a new chat history that includes:
       - All messages up to the insertion point.
       - Any rejection messages (pairs of tool call and tool call result messages).
       - Any tool call messages for confirmed or modified tool calls, including user messages explaining modifications.

    :param chat_history: The current chat history.
    :param rejection_messages: Chat messages to add for rejected tool calls (pairs of tool call and tool call result
        messages).
    :param tool_call_and_explanation_messages: Tool call messages for confirmed or modified tool calls, which may
        include user messages explaining modifications.
    :returns:
        The updated chat history.
    """
    user_indices = [i for i, message in enumerate(chat_history) if message.is_from("user")]
    tool_indices = [i for i, message in enumerate(chat_history) if message.is_from("tool")]

    last_user_idx = max(user_indices) if user_indices else -1
    last_tool_idx = max(tool_indices) if tool_indices else -1

    insertion_point = max(last_user_idx, last_tool_idx)

    new_chat_history = chat_history[: insertion_point + 1] + rejection_messages + tool_call_and_explanation_messages
    return new_chat_history
