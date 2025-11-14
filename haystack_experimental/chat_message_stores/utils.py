# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses import ChatMessage


def get_last_k_messages(messages: list[ChatMessage], last_k: int) -> list[ChatMessage]:
    """
    Get the last_k messages from the list of messages while ensuring the chat history remains valid.

    By valid we mean that if there are ToolCalls in the chat history, we want to ensure that we do not slice
    in a way that a ToolCall is present without its corresponding ToolOutput.

    We handle this by treating ToolCalls and its corresponding ToolOutput(s) as a single unit when slicing the chat
    history.

    :param messages:
        The list of chat messages.
    :param last_k:
        The number of last messages to retrieve.
    :returns:
        The sliced list of chat messages.
    """
    # If ToolCalls are present we try to keep pairs of ToolCalls + ToolOutputs together to ensure a valid
    # chat history. We only slice at indices where the number of ToolCalls matches the number of ToolOutputs.
    has_tool_calls = any(msg.tool_call is not None for msg in messages)
    if has_tool_calls:
        valid_start_indices = []
        for start_idx in range(len(messages)):
            sliced_messages = messages[start_idx:]
            num_tool_calls_in_messages = sum(len(msg.tool_calls) for msg in sliced_messages)
            num_tool_outputs_in_messages = sum(len(msg.tool_call_results) for msg in sliced_messages)
            if num_tool_calls_in_messages == num_tool_outputs_in_messages:
                valid_start_indices.append(start_idx)

        # If no valid start index is found we return the entire history
        # This should only occur if the chat history consists of only tool call messages
        if not valid_start_indices:
            valid_start_indices = [0]

        new_start_index = valid_start_indices[-last_k] if len(valid_start_indices) >= last_k else 0
        messages = messages[new_start_index:]
    else:
        messages = messages[-last_k:]
    return messages
