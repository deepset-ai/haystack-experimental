# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from base64 import b64encode

import pytest

from haystack_experimental.dataclasses import ChatMessage, ChatRole, ToolCall, ToolCallResult, TextContent, \
    MediaContent, ByteStream


def test_tool_call_init():
    tc = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    assert tc.id == "123"
    assert tc.tool_name == "mytool"
    assert tc.arguments == {"a": 1}

def test_tool_call_result_init():
    tcr = ToolCallResult(result="result", origin=ToolCall(id="123", tool_name="mytool", arguments={"a": 1}), error=True)
    assert tcr.result == "result"
    assert tcr.origin == ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    assert tcr.error

def test_text_content_init():
    tc = TextContent(text="Hello")
    assert tc.text == "Hello"

def test_media_content_init():
    mc = MediaContent(media=ByteStream(data=b"media data", mime_type="image/png"))
    assert mc.media.data == b"media data"
    assert mc.media.mime_type == "image/png"

def test_from_assistant_with_valid_content():
    text = "Hello, how can I assist you?"
    message = ChatMessage.from_assistant(text)

    assert message._role == ChatRole.ASSISTANT
    assert message._content == [TextContent(text)]

    assert message.text == text
    assert message.texts == [text]

    assert not message.media
    assert not message.tool_calls
    assert not message.tool_call
    assert not message.tool_call_results
    assert not message.tool_call_result

def test_from_assistant_with_tool_calls():
    tool_calls = [ToolCall(id="123", tool_name="mytool", arguments={"a": 1}),
                  ToolCall(id="456", tool_name="mytool2", arguments={"b": 2})]

    message = ChatMessage.from_assistant(tool_calls=tool_calls)

    assert message.role == ChatRole.ASSISTANT
    assert message._content == tool_calls

    assert message.tool_calls == tool_calls
    assert message.tool_call == tool_calls[0]

    assert not message.texts
    assert not message.text
    assert not message.media
    assert not message.tool_call_results
    assert not message.tool_call_result


def test_from_user_with_valid_content():
    text = "I have a question."
    message = ChatMessage.from_user(text=text)

    assert message.role == ChatRole.USER
    assert message._content == [TextContent(text)]

    assert message.text == text
    assert message.texts == [text]

    assert not message.media
    assert not message.tool_calls
    assert not message.tool_call
    assert not message.tool_call_results
    assert not message.tool_call_result

def test_from_user_with_media():
    text = "This is a multimodal message!"
    media = [ByteStream(data=b"media data", mime_type="image/png")]
    message = ChatMessage.from_user(text=text, media=media)

    assert message.role == ChatRole.USER
    assert message._content == [TextContent(text="This is a multimodal message!"), MediaContent(media[0])]

    assert message.text == text
    assert message.texts == [text]
    assert message.media == media

    assert not message.tool_calls
    assert not message.tool_call
    assert not message.tool_call_results
    assert not message.tool_call_result

def test_from_system_with_valid_content():
    text = "I have a question."
    message = ChatMessage.from_system(text=text)

    assert message.role == ChatRole.SYSTEM
    assert message._content == [TextContent(text)]

    assert message.text == text
    assert message.texts == [text]

    assert not message.media
    assert not message.tool_calls
    assert not message.tool_call
    assert not message.tool_call_results
    assert not message.tool_call_result

def test_from_tool_with_valid_content():
    tool_result = "Tool result"
    origin = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    message = ChatMessage.from_tool(tool_result, origin, error=False)

    tcr = ToolCallResult(result=tool_result, origin=origin, error=False)

    assert message._content == [tcr]
    assert message.role == ChatRole.TOOL

    assert message.tool_call_result == tcr
    assert message.tool_call_results == [tcr]

    assert not message.media
    assert not message.tool_calls
    assert not message.tool_call
    assert not message.texts
    assert not message.text

def test_multiple_text_segments():
    texts = [TextContent(text="Hello"), TextContent(text="World")]
    message = ChatMessage(_role=ChatRole.USER, _content=texts)

    assert message.texts == ["Hello", "World"]
    assert len(message) == 2

def test_mixed_content():
    content = [
        TextContent(text="Hello"),
        ToolCall(id="123", tool_name="mytool", arguments={"a": 1}),
    ]

    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=content)

    assert len(message) == 2
    assert message.texts == ["Hello"]
    assert message.text == "Hello"

    assert message.tool_calls == [content[1]]
    assert message.tool_call == content[1]

def test_serde():
    # the following message is created just for testing purposes and does not make sense in a real use case

    role = ChatRole.ASSISTANT

    text_content = TextContent(text="Hello")
    media_content = MediaContent(media=ByteStream(data=b"media_data", mime_type="image/png"))
    tool_call = ToolCall(id="123", tool_name="mytool", arguments={"a": 1})
    tool_call_result = ToolCallResult(result="result", origin=tool_call, error=False)
    meta = {"some": "info"}

    message = ChatMessage(
        _role=role,
        _content=[text_content, media_content, tool_call, tool_call_result],
        _name="my_message",
        _meta=meta,
    )

    serialized_message = message.to_dict()
    assert serialized_message == {
        "_content": [
            {"text": "Hello"},
            {"media": {"data": b64encode(b"media_data").decode(), "meta": {}, "mime_type": "image/png"}},
            {"tool_call": {"id": "123", "tool_name": "mytool", "arguments": {"a": 1}}},
            {
                "tool_call_result": {
                    "result": "result",
                    "error": False,
                    "origin": {"id": "123", "tool_name": "mytool", "arguments": {"a": 1}},
                }
            },
        ],
        "_role": "assistant",
        "_name": "my_message",
        "_meta": {"some": "info"},
    }

    deserialized_message = ChatMessage.from_dict(serialized_message)
    assert deserialized_message == message

def test_to_dict_with_invalid_content_type():
    text_content = TextContent(text="Hello")
    invalid_content = "invalid"

    message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[text_content, invalid_content])

    with pytest.raises(TypeError):
        message.to_dict()


def test_from_dict_with_invalid_content_type():
    data = {"_role": "assistant", "_content": [{"text": "Hello"}, "invalid"]}
    with pytest.raises(ValueError):
        ChatMessage.from_dict(data)

    data = {"_role": "assistant", "_content": [{"text": "Hello"}, {"invalid": "invalid"}]}
    with pytest.raises(ValueError):
        ChatMessage.from_dict(data)
