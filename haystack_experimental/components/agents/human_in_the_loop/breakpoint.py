# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentSnapshot

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ToolExecutionDecision


# TODO This potentially could live in hayhooks instead. At best it's a utility function that is only used in the
#      BreakpointConfirmationStrategy example and could be used in hayhooks to enable human-in-the-loop for agents.
def _get_tool_calls_and_descriptions(agent_snapshot: AgentSnapshot) -> tuple[list[dict], dict[str, str]]:
    # Create the list of tool calls to send
    serialized_tool_call_messages = agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["messages"]
    tool_call_messages = [ChatMessage.from_dict(m) for m in serialized_tool_call_messages]
    tool_calls = []
    for msg in tool_call_messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    serialized_tcs = [tc.to_dict() for tc in tool_calls]
    # TODO The arguments in serialized_tool_calls are not fully correct. Missing the injection from inputs_from_state.

    # Create the dict of tool descriptions to send
    serialized_tools = agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["tools"]
    tool_descriptions = {t["data"]["name"]: t["data"]["description"] for t in serialized_tools}
    return serialized_tcs, tool_descriptions


# TODO Probably move the two below functions to human_in_the_loop/strategies.py since they don't have anything to do
#      with breakpoints specifically.
def _apply_tool_execution_decisions(
    chat_history: list[ChatMessage],
    tool_call_messages: list[ChatMessage],
    tool_execution_decisions: list[ToolExecutionDecision]
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """
    Apply the tool execution decisions to the tool call messages and update the provided chat history.

    :param chat_history: The current chat history.
    :param tool_call_messages: The tool call messages to apply the decisions to.
    :param tool_execution_decisions: The tool execution decisions to apply.
    :returns:
        A tuple containing the updated chat history and the list of tool call messages that were executed (i.e.
        confirmed or modified).
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

    # Update the chat history with rejection messages and new tool call messages
    new_chat_history = _update_chat_history(chat_history, rejection_messages, new_tool_call_messages)

    return new_chat_history, new_tool_call_messages


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
