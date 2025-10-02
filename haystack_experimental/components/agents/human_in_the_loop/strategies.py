# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from haystack.components.agents.state import State, replace_values
from haystack.components.tools.tool_invoker import ToolInvoker
from haystack.core.serialization import default_from_dict, default_to_dict, import_class_by_name
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool

from haystack_experimental.components.agents.human_in_the_loop import (
    ConfirmationPolicy,
    ConfirmationStrategy,
    ConfirmationUI,
    ToolExecutionDecision,
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


def prepare_tool_args(
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


def _handle_confirmation_strategies(
    *,
    confirmation_strategies: dict[str, ConfirmationStrategy],
    messages_with_tool_calls: list[ChatMessage],
    execution_context: "_ExecutionContext",
) -> tuple[list[ChatMessage], "_ExecutionContext"]:
    """
    Handle tool execution confirmation strategies for tool calls in the provided messages.

    :param confirmation_strategies: Mapping of tool names to their corresponding confirmation strategies
    :param messages_with_tool_calls: Messages containing tool calls to process
    :param execution_context: The current execution context containing state and inputs
    :returns: Tuple of modified messages with confirmed tool calls and tool call result messages
    """
    state = execution_context.state
    tools_with_names = {tool.name: tool for tool in execution_context.tool_invoker_inputs["tools"]}
    existing_teds = execution_context.tool_execution_decisions if execution_context.tool_execution_decisions else []

    teds = []
    for message in messages_with_tool_calls:
        if not message.tool_calls:
            continue

        # confirmed_tool_calls = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.tool_name
            tool_to_invoke = tools_with_names[tool_name]

            # Prepare final tool args
            final_args = prepare_tool_args(
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

    rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
        tool_call_messages=messages_with_tool_calls,
        tool_execution_decisions=teds,
    )

    # Update the chat history with rejection messages and new tool call messages
    new_chat_history = _update_chat_history(state.get("messages"), rejection_messages, modified_tool_call_messages)

    # Update chat history in state
    state.set(key="messages", value=new_chat_history, handler_override=replace_values)

    return modified_tool_call_messages, execution_context


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
        - A list of modified tool call messages for confirmed or modified tool calls.
    """
    new_tool_call_messages = []
    rejection_messages = []
    for chat_msg in tool_call_messages:
        if not chat_msg.tool_calls:
            continue

        new_tool_calls = []
        for tc in chat_msg.tool_calls:
            ted = next(
                ted for ted in tool_execution_decisions if (ted.tool_id == tc.id or ted.tool_name == tc.tool_name)
            )
            if ted.execute:
                # Covers confirm and modify cases
                if tc.arguments != ted.final_tool_params:
                    # In the modify case we add a user message explaining the modification otherwise the LLM won't know
                    # why the tool parameters changed and will likely just try and call the tool again with the
                    # original parameters.
                    new_tool_call_messages.append(
                        ChatMessage.from_user(
                            text=(
                                f"The parameters for tool '{tc.tool_name}' were updated by the user to:\n"
                                f"{ted.final_tool_params}"
                            )
                        )
                    )
                new_tool_calls.append(replace(tc, arguments=ted.final_tool_params))
            else:
                # Reject case
                # We create a tool call and tool call result message pair to put into the chat history of State
                rejection_messages.append(
                    # We can't use dataclasses.replace, so we use from_assistant to create a new message
                    ChatMessage.from_assistant(
                        text=chat_msg.text,
                        meta=chat_msg.meta,
                        name=chat_msg.name,
                        tool_calls=[tc],
                        reasoning=chat_msg.reasoning,
                    )
                )
                rejection_messages.append(
                    ChatMessage.from_tool(
                        tool_result=ted.feedback or "",
                        origin=tc,
                        error=True,
                    )
                )

        # Only add the tool call message if there are any tool calls left (i.e. not all were rejected)
        if new_tool_calls:
            new_tool_call_messages.append(
                ChatMessage.from_assistant(
                    text=chat_msg.text,
                    meta=chat_msg.meta,
                    name=chat_msg.name,
                    tool_calls=new_tool_calls,
                    reasoning=chat_msg.reasoning,
                )
            )

    return rejection_messages, new_tool_call_messages


def _update_chat_history(
    chat_history: list[ChatMessage], rejection_messages: list[ChatMessage], tool_call_messages: list[ChatMessage]
):
    """
    Update the chat history to include rejection messages and tool call messages at the appropriate positions.

    Steps:
    1. Identify the last user message and the last tool message in the current chat history.
    2. Determine the last relevant message index, which is the last of the two identified messages.
    3. Create a new chat history that includes:
       - All messages up to and including the last relevant message.
       - Any rejection messages (pairs of tool call and tool call result messages).
       - Any tool call messages for confirmed or modified tool calls.

    :param chat_history: The current chat history.
    :param rejection_messages: Chat messages to add for rejected tool calls. These should be pairs of tool call and
        tool call result messages.
    :param tool_call_messages: The tool call messages for confirmed or modified tool calls that should be added to the
        end of the chat history.
    :returns:
        The updated chat history.
    """
    last_user_msg_idx = max(i for i, m in enumerate(chat_history) if m.is_from("user"))
    last_tool_msg_idx = max(
        [i for i, m in enumerate(chat_history) if m.is_from("tool")] + [-1]
    )  # -1 in case there are no tool messages yet
    last_relevant_msg_idx = max(last_user_msg_idx, last_tool_msg_idx)
    # We take everything up to and including the last relevant message, then add any additional messages
    new_chat_history = chat_history[: last_relevant_msg_idx + 1] + rejection_messages + tool_call_messages
    return new_chat_history
