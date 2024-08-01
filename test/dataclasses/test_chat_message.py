# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack_experimental.dataclasses import ChatMessage, ChatRole, ToolCall

def test_tool_call_init():
    tc = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    assert tc.id == "123"
    assert tc.tool_name == "mytool"
    assert tc.arguments == {"a": 1}

def test_from_assistant_with_valid_content():
    content = "Hello, how can I assist you?"
    message = ChatMessage.from_assistant(content)
    assert message.content == content
    assert message.role == ChatRole.ASSISTANT

def test_from_assistant_with_tool_calls():
    content =""
    tool_calls = [ToolCall(id="123", tool_name="mytool", arguments={"a": 1}),
                  ToolCall(id="456", tool_name="mytool2", arguments={"b": 2})]

    message = ChatMessage.from_assistant(content=content, tool_calls=tool_calls)

    assert message.role == ChatRole.ASSISTANT
    assert message.content == content
    assert message.tool_calls == tool_calls


def test_from_user_with_valid_content():
    content = "I have a question."
    message = ChatMessage.from_user(content)
    assert message.content == content
    assert message.role == ChatRole.USER


def test_from_system_with_valid_content():
    content = "System message."
    message = ChatMessage.from_system(content)
    assert message.content == content
    assert message.role == ChatRole.SYSTEM

def test_from_tool_with_valid_content():
    content = "Tool message."
    tool_call_id = "123"
    message = ChatMessage.from_tool(content, tool_call_id)

    assert message.content == content
    assert message.role == ChatRole.TOOL
    assert message.tool_call_id == tool_call_id


def test_with_empty_content():
    message = ChatMessage.from_user("")
    assert message.content == ""


def test_to_dict():
    message = ChatMessage.from_user("content")
    message.meta["some"] = "some"

    assert message.to_dict() == {'content': 'content', 'role': 'user', 'tool_call_id': None, 'tool_calls': [], 'meta': {'some': 'some'}}


def test_from_dict():
    assert ChatMessage.from_dict(data={"content": "text", "role": "user"}) == ChatMessage.from_user(content="text")


def test_from_dict_with_meta():
    assert ChatMessage.from_dict(
        data={"content": "text", "role": "assistant", "meta": {"something": "something"}}
    ) == ChatMessage.from_assistant(content="text", meta={"something": "something"})

def test_serde_with_tool_calls():
    tool_call = ToolCall(id="123", tool_name="tool", arguments={"a": 1})
    message = ChatMessage.from_assistant("a message", tool_calls=[tool_call])

    serialized_message = message.to_dict()

    assert serialized_message=={'content': 'a message', 'role': 'assistant', 'tool_call_id': None, 'tool_calls': [{'id': '123', 'tool_name': 'tool', 'arguments': {'a': 1}}], 'meta': {}}

    deserialized_message = ChatMessage.from_dict(serialized_message)
    assert deserialized_message == message
