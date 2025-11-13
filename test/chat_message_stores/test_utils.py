# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ToolCall

from haystack_experimental.chat_message_stores.utils import get_last_k_messages


class TestGetLastKMessages:
    @pytest.mark.parametrize("last_k, start_idx", [(1, 4), (2, 2), (3, 1), (4, 0), (5, 0)])
    def test_last_k_agent_history(self, last_k, start_idx):
        tool_call = ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        # System, User, ToolCall, ToolOutput, Assistant
        messages = [
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="It's sunny in Paris.", origin=tool_call),
            ChatMessage.from_assistant("The weather in Paris is sunny."),
        ]

        last_k_messages = get_last_k_messages(messages, last_k)

        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k, start_idx", [(1, 5), (2, 2), (3, 1), (4, 0), (5, 0), (6, 0)])
    def test_last_k_agent_history_with_two_tool_calls(self, last_k, start_idx):
        tool_call_1 = ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        tool_call_2 = ToolCall(tool_name="get_time", arguments={"location": "Paris"})
        # System, User, 2 ToolCalls, ToolOutput, ToolOutput, Assistant
        messages = [
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call_1, tool_call_2]),
            ChatMessage.from_tool(tool_result="It's sunny in Paris.", origin=tool_call_1),
            ChatMessage.from_tool(tool_result="It's 3 PM in Paris.", origin=tool_call_2),
            ChatMessage.from_assistant("The weather in Paris is sunny and it's 3 PM there."),
        ]

        last_k_messages = get_last_k_messages(messages, last_k)
        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k,start_idx", [(1, 5), (2, 2), (3, 1), (4, 0), (5, 0), (6, 0)])
    def test_last_k_agent_history_with_intervening_user_message(self, last_k, start_idx):
        """Relevant for simulating a human-in-the-loop scenario where the user modifies parameters"""
        tool_call = ToolCall(tool_name="get_weather", arguments={"location": "Paris"}, id="tc_1")
        modified_tool_call = ToolCall(tool_name="get_weather", arguments={"location": "London"}, id="tc_1")
        # System, User, ToolCall, User, ToolOutput, Assistant
        messages = [
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_user("The user modified the tool parameters to {'location': 'London'}."),
            ChatMessage.from_tool(tool_result="It's sunny in London.", origin=modified_tool_call),
            ChatMessage.from_assistant("The weather in London is sunny."),
        ]

        last_k_messages = get_last_k_messages(messages, last_k=last_k)
        assert last_k_messages == messages[start_idx:]