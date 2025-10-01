# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from haystack.components.agents.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentSnapshot

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ToolExecutionDecision


def get_tool_calls_and_descriptions(agent_snapshot: AgentSnapshot) -> tuple[list[dict], dict[str, str]]:
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


def _update_state_and_tool_call_messages_with_tool_execution_decisions(
    state: State, tool_call_messages: list[ChatMessage], tool_execution_decisions: list[ToolExecutionDecision]
) -> tuple[State, list[ChatMessage]]:
    new_tool_call_messages = []
    additional_state_messages = []
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
                additional_state_messages.append(
                    # We can't use dataclasses.replace, so we use from_assistant to create a new message
                    ChatMessage.from_assistant(
                        text=chat_msg.text,
                        meta=chat_msg.meta,
                        name=chat_msg.name,
                        tool_calls=[tc],
                        reasoning=chat_msg.reasoning,
                    )
                )
                additional_state_messages.append(
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

    # Modify the chat history in state to handle the rejection cases
    # 1. Move the tool call message and tool call result message pairs to right before last ToolCallResult or user
    #    message whichever is later in the chat history
    # 2. Put all outstanding tool call messages (i.e. the ones that were confirmed or modified) at the end of the chat
    #    history
    chat_history = state.get("messages")
    last_user_msg_idx = max(i for i, m in enumerate(chat_history) if m.is_from("user"))
    last_tool_msg_idx = max(
        [i for i, m in enumerate(chat_history) if m.is_from("tool")] + [-1]
    )  # -1 in case there are no tool messages yet
    last_relevant_msg_idx = max(last_user_msg_idx, last_tool_msg_idx)
    # We take everything up to and including the last relevant message, then add any additional messages
    new_chat_history = chat_history[: last_relevant_msg_idx + 1] + additional_state_messages + new_tool_call_messages
    state.set(key="messages", value=new_chat_history, handler_override=replace_values)

    return state, new_tool_call_messages
