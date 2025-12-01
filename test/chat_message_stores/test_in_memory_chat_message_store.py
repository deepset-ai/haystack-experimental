# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage, ToolCall

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore


@pytest.fixture
def store():
    msg_store = InMemoryChatMessageStore()
    yield msg_store
    msg_store.delete_all_messages()


class TestInMemoryChatMessageStore:

    def test_init(self, store):
        """
        Test that the InMemoryChatMessageStore can be initialized and that it works as expected.
        """
        assert store.count_messages(chat_history_id="test") == 0
        assert store.retrieve_messages(chat_history_id="test") == []
        assert store.write_messages(chat_history_id="test", messages=[]) == 0
        assert not store.delete_messages(chat_history_id="test")

    def test_to_dict(self):
        """
        Test that the InMemoryChatMessageStore can be serialized to a dictionary.
        """
        store = InMemoryChatMessageStore()
        assert store.to_dict() == {
            "init_parameters": {
                "skip_system_messages": True,
                "last_k": 10
            },
            "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"
        }

    def test_from_dict(self):
        """
        Test that the InMemoryChatMessageStore can be deserialized from a dictionary.
        """
        data = {
            "init_parameters": {
                "skip_system_messages": True,
                "last_k": 10
            },
            "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"
        }
        store = InMemoryChatMessageStore.from_dict(data)
        assert store.to_dict() == data

    def test_count_messages(self, store):
        """
        Test that the InMemoryChatMessageStore can count the number of messages in the store correctly.
        """
        assert store.count_messages(chat_history_id="test") == 0
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        assert store.count_messages(chat_history_id="test") == 1
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")])
        assert store.count_messages(chat_history_id="test") == 2
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?")])
        assert store.count_messages(chat_history_id="test") == 3

    def test_retrieve(self, store):
        """
        Test that the InMemoryChatMessageStore can retrieve all messages from the store correctly.
        """
        assert store.retrieve_messages(chat_history_id="test") == []
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        assert store.retrieve_messages(chat_history_id="test") == [
            ChatMessage.from_user("Hello, how can I help you?", meta={"chat_message_id": "0"}),
        ]
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")])
        assert store.retrieve_messages(chat_history_id="test") == [
            ChatMessage.from_user("Hello, how can I help you?", meta={"chat_message_id": "0"}),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?", meta={"chat_message_id": "1"}),
        ]
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?")])
        assert store.retrieve_messages(chat_history_id="test") == [
            ChatMessage.from_user("Hello, how can I help you?", meta={"chat_message_id": "0"}),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?", meta={"chat_message_id": "2"}),
        ]

    def test_delete_messages(self, store):
        """
        Test that the InMemoryChatMessageStore can delete all messages from the store correctly.
        """
        assert store.count_messages(chat_history_id="test") == 0
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        assert store.count_messages(chat_history_id="test") == 1
        store.delete_messages(chat_history_id="test")
        assert store.count_messages(chat_history_id="test") == 0
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")])
        store.write_messages(chat_history_id="test", messages=[ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?")])
        assert store.count_messages(chat_history_id="test") == 2
        store.delete_messages(chat_history_id="test")
        assert store.count_messages(chat_history_id="test") == 0


class TestGetLastKMessages:
    @pytest.mark.parametrize("last_k, start_idx", [(1, 2), (2, 0), (3, 0)])
    def test_last_k_chat_history(self, last_k, start_idx):
        tool_call = ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        # This is a history that contains two pairs of User - Assistant messages.
        messages = [
            # Pair 1
            ChatMessage.from_user("Hello!"),
            ChatMessage.from_assistant("Hi! How can I assist you today?"),
            # Pair 2 w/ one tool call
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="It's sunny in Paris.", origin=tool_call),
            ChatMessage.from_assistant("The weather in Paris is sunny."),
        ]

        last_k_messages = InMemoryChatMessageStore._get_last_k_messages(messages, last_k)

        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k, start_idx", [(1, 3), (2, 1), (3, 0)])
    def test_last_k_chat_history_with_system_prompt(self, last_k, start_idx):
        tool_call = ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        # This is a history that contains three pairs of User - Assistant messages.
        # The system prompt counts as a full pair.
        messages = [
            # "Pair 1"
            ChatMessage.from_system("You are a helpful assistant."),
            # Pair 2
            ChatMessage.from_user("Hello!"),
            ChatMessage.from_assistant("Hi! How can I assist you today?"),
            # Pair 3 w/ one tool call
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="It's sunny in Paris.", origin=tool_call),
            ChatMessage.from_assistant("The weather in Paris is sunny."),
        ]

        last_k_messages = InMemoryChatMessageStore._get_last_k_messages(messages, last_k)

        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k, start_idx", [(1, 5), (2, 0)])
    def test_last_k_chat_history_with_two_tool_calls(self, last_k, start_idx):
        tool_call_1 = ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        tool_call_2 = ToolCall(tool_name="get_time", arguments={"location": "Paris"})
        messages = [
            # Pair 1 w/ two tool calls
            # System, User, 2 ToolCalls, ToolOutput, ToolOutput, Assistant
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call_1, tool_call_2]),
            ChatMessage.from_tool(tool_result="It's sunny in Paris.", origin=tool_call_1),
            ChatMessage.from_tool(tool_result="It's 3 PM in Paris.", origin=tool_call_2),
            ChatMessage.from_assistant("The weather in Paris is sunny and it's 3 PM there."),
            # Pair 2
            ChatMessage.from_user("Thank you!"),
            ChatMessage.from_assistant("You're welcome! Let me know if you have any other questions."),
        ]

        last_k_messages = InMemoryChatMessageStore._get_last_k_messages(messages, last_k)
        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k,start_idx", [(1, 2), (2, 0)])
    def test_last_k_chat_history_with_one_modified_tool_calls(self, last_k, start_idx):
        """Relevant for simulating a human-in-the-loop scenario where the user modifies parameters"""
        # "Original" tool call had params {"location": "Paris"}
        modified_tool_call = ToolCall(tool_name="get_weather", arguments={"location": "London"}, id="tc_1")
        messages = [
            # Pair 1
            ChatMessage.from_user("Hello!"),
            ChatMessage.from_assistant("Hi! How can I assist you today?"),
            # Pair 2 w/ one tool call and an intervening user message
            # System, User, ToolCall, User, ToolOutput, Assistant
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_user("The user modified the tool parameters to {'location': 'London'}."),
            ChatMessage.from_assistant(tool_calls=[modified_tool_call]),
            ChatMessage.from_tool(tool_result="It's sunny in London.", origin=modified_tool_call),
            ChatMessage.from_assistant("The weather in London is sunny."),
        ]

        last_k_messages = InMemoryChatMessageStore._get_last_k_messages(messages, last_k=last_k)
        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k,start_idx", [(1, 2), (2, 0)])
    def test_last_k_chat_history_with_two_modified_tool_calls(self, last_k, start_idx):
        """Relevant for simulating a human-in-the-loop scenario where the user modifies parameters"""
        # "Original" tool calls had params {"location": "Paris"} and {"location": "Berlin"}
        modified_tool_call_1 = ToolCall(tool_name="get_weather", arguments={"location": "London"}, id="tc_1")
        modified_tool_call_2 = ToolCall(tool_name="get_time", arguments={"location": "Brussels"}, id="tc_2")
        messages = [
            # Pair 1
            ChatMessage.from_user("Hello!"),
            ChatMessage.from_assistant("Hi! How can I assist you today?"),
            # Pair 2 w/ two tool calls and intervening user messages
            ChatMessage.from_user("What is the weather in Paris and the time in Berlin?"),
            ChatMessage.from_user(
                "The parameters for tool 'get_weather' were updated by the user to:\n{'location': 'London'}."
            ),
            ChatMessage.from_user(
                "The parameters for tool 'get_time' were updated by the user to:\n{'location': 'Brussels'}."
            ),
            ChatMessage.from_assistant(tool_calls=[modified_tool_call_1, modified_tool_call_2]),
            ChatMessage.from_tool(tool_result="It's sunny in London.", origin=modified_tool_call_1),
            ChatMessage.from_tool(tool_result="It's 13:00 in Brussels.", origin=modified_tool_call_2),
            ChatMessage.from_assistant("The weather in London is sunny and it's 13:00 in Brussels."),
        ]

        last_k_messages = InMemoryChatMessageStore._get_last_k_messages(messages, last_k=last_k)
        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k,start_idx", [(1, 2), (2, 0)])
    def test_last_k_chat_history_with_two_tool_calls_one_modified_one_rejected(self, last_k, start_idx):
        """Relevant for simulating a human-in-the-loop scenario where the user modifies parameters"""
        modified_tool_call_1 = ToolCall(tool_name="get_weather", arguments={"location": "London"}, id="tc_1")
        tool_call_2 = ToolCall(tool_name="get_time", arguments={"location": "Berlin"}, id="tc_2")
        messages = [
            # Pair 1
            ChatMessage.from_user("Hello!"),
            ChatMessage.from_assistant("Hi! How can I assist you today?"),
            # Pair 2 w/ two tool calls and intervening user messages
            ChatMessage.from_user("What is the weather in Paris and the time in Berlin?"),
            ChatMessage.from_assistant(tool_calls=[tool_call_2]),  # Tool Call 2 was rejected
            ChatMessage.from_tool(
                tool_result="The tool execution for 'tool2' was rejected by the user. With feedback: Not needed",
                origin=tool_call_2
            ),
            ChatMessage.from_user( # Tool Call 1 was modified
                "The parameters for tool 'get_weather' were updated by the user to:\n{'location': 'London'}."
            ),
            ChatMessage.from_assistant(tool_calls=[modified_tool_call_1]),
            ChatMessage.from_tool(tool_result="It's sunny in London.", origin=modified_tool_call_1),
            ChatMessage.from_assistant("The weather in London is sunny."),
        ]

        last_k_messages = InMemoryChatMessageStore._get_last_k_messages(messages, last_k=last_k)
        assert last_k_messages == messages[start_idx:]