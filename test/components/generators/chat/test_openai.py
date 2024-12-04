# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from base64 import b64encode
from unittest.mock import MagicMock, patch
import pytest

from typing import Iterator
import logging
import os
import json
from datetime import datetime

from openai import OpenAIError
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function
from openai.types.chat import chat_completion_chunk
from openai import Stream

from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import StreamingChunk
from haystack.utils.auth import Secret
from haystack_experimental.dataclasses import (
    ChatMessage,
    Tool,
    ToolCall,
    ChatRole,
    TextContent,
    ByteStream,
)
from haystack_experimental.components.generators.chat.openai import (
    OpenAIChatGenerator,
    _convert_message_to_openai_format,
)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


class MockStream(Stream[ChatCompletionChunk]):
    def __init__(self, mock_chunk: ChatCompletionChunk, client=None, *args, **kwargs):
        client = client or MagicMock()
        super().__init__(client=client, *args, **kwargs)
        self.mock_chunk = mock_chunk

    def __stream__(self) -> Iterator[ChatCompletionChunk]:
        # Yielding only one ChatCompletionChunk object
        yield self.mock_chunk


@pytest.fixture
def mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(
                        content="Hello", role="assistant"
                    ),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        mock_chat_completion_create.return_value = MockStream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="gpt-4",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(
                        content="Hello world!", role="assistant"
                    ),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture
def mock_chat_completion_chunk_with_tools():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    with patch(
        "openai.resources.chat.completions.Completions.create"
    ) as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="tool_calls",
                    logprobs=None,
                    index=0,
                    delta=chat_completion_chunk.ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            chat_completion_chunk.ChoiceDeltaToolCall(
                                index=0,
                                id="123",
                                type="function",
                                function=chat_completion_chunk.ChoiceDeltaToolCallFunction(
                                    name="weather", arguments='{"city": "Paris"}'
                                ),
                            )
                        ],
                    ),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        mock_chat_completion_create.return_value = MockStream(
            completion, cast_to=None, response=None, client=None
        )
        yield mock_chat_completion_create


@pytest.fixture
def tools():
    tool_parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=lambda x: x,
    )

    return [tool]


class TestOpenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.client.timeout == 30
        assert component.client.max_retries == 5
        assert component.tools is None
        assert not component.tools_strict

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError):
            OpenAIChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            OpenAIChatGenerator(tools=duplicate_tools)

    def test_init_with_parameters(self, monkeypatch):
        tool = Tool(
            name="name",
            description="description",
            parameters={"x": {"type": "string"}},
            function=lambda x: x,
        )

        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=40.0,
            max_retries=1,
            tools=[tool],
            tools_strict=True,
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }
        assert component.client.timeout == 40.0
        assert component.client.max_retries == 1
        assert component.tools == [tool]
        assert component.tools_strict

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }
        assert component.client.timeout == 100.0
        assert component.client.max_retries == 10

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["OPENAI_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "gpt-4o-mini",
                "organization": None,
                "streaming_callback": None,
                "api_base_url": None,
                "generation_kwargs": {},
                "tools": None,
                "tools_strict": False,
                "max_retries": None,
                "timeout": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        tool = Tool(
            name="name",
            description="description",
            parameters={"x": {"type": "string"}},
            function=print,
        )

        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = OpenAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
            tools_strict=True,
            max_retries=10,
            timeout=100.0,
        )
        data = component.to_dict()

        assert data == {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "organization": None,
                "api_base_url": "test-base-url",
                "max_retries": 10,
                "timeout": 100.0,
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
                "tools": [
                    {
                        "description": "description",
                        "function": "builtins.print",
                        "name": "name",
                        "parameters": {
                            "x": {
                                "type": "string",
                            },
                        },
                    },
                ],
                "tools_strict": True,
            },
        }

    def test_to_dict_with_lambda_streaming_callback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        component = OpenAIChatGenerator(
            model="gpt-4o-mini",
            streaming_callback=lambda x: x,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["OPENAI_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "gpt-4o-mini",
                "organization": None,
                "api_base_url": "test-base-url",
                "max_retries": None,
                "timeout": None,
                "streaming_callback": "test_openai.<lambda>",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
                "tools": None,
                "tools_strict": False,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["OPENAI_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "gpt-4o-mini",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "max_retries": 10,
                "timeout": 100.0,
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
                "tools": [
                    {
                        "description": "description",
                        "function": "builtins.print",
                        "name": "name",
                        "parameters": {
                            "x": {
                                "type": "string",
                            },
                        },
                    },
                ],
                "tools_strict": True,
            },
        }
        component = OpenAIChatGenerator.from_dict(data)

        assert isinstance(component, OpenAIChatGenerator)
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }
        assert component.api_key == Secret.from_env_var("OPENAI_API_KEY")
        assert component.tools == [
            Tool(
                name="name",
                description="description",
                parameters={"x": {"type": "string"}},
                function=print,
            )
        ]
        assert component.tools_strict
        assert component.client.timeout == 100.0
        assert component.client.max_retries == 10

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        data = {
            "type": "haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["OPENAI_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "gpt-4",
                "organization": None,
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
            },
        }
        with pytest.raises(ValueError):
            OpenAIChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion):
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            generation_kwargs={"max_tokens": 10, "temperature": 0.5},
        )
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params_streaming(self, chat_messages, mock_chat_completion_chunk):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            streaming_callback=streaming_callback,
        )
        response = component.run(chat_messages)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].text  # see mock_chat_completion_chunk

    def test_run_with_streaming_callback_in_run_method(
        self, chat_messages, mock_chat_completion_chunk
    ):
        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages, streaming_callback=streaming_callback)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]
        assert "Hello" in response["replies"][0].text  # see mock_chat_completion_chunk

    def test_check_abnormal_completions(self, caplog):
        caplog.set_level(logging.INFO)
        component = OpenAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        messages = [
            ChatMessage.from_assistant(
                "",
                meta={
                    "finish_reason": "content_filter" if i % 2 == 0 else "length",
                    "index": i,
                },
            )
            for i, _ in enumerate(range(4))
        ]

        for m in messages:
            component._check_finish_reason(m.meta)

        # check truncation warning
        message_template = (
            "The completion for index {index} has been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions."
        )

        for index in [1, 3]:
            assert caplog.records[index].message == message_template.format(index=index)

        # check content filter warning
        message_template = "The completion for index {index} has been truncated due to the content filter."
        for index in [0, 2]:
            assert caplog.records[index].message == message_template.format(index=index)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIChatGenerator(generation_kwargs={"n": 1})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenAIChatGenerator(generation_kwargs={"n": 1})
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = OpenAIChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_wrong_model_async(self, chat_messages):
        component = OpenAIChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            await component.run_async(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = OpenAIChatGenerator(streaming_callback=callback)
        results = component.run(
            [ChatMessage.from_user("What's the capital of France?")]
        )

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_streaming_async(self):
        counter = 0
        responses = ""

        async def callback(chunk: StreamingChunk):
            nonlocal counter
            nonlocal responses
            counter += 1
            responses += chunk.content if chunk.content else ""

        component = OpenAIChatGenerator(streaming_callback=callback)
        results = await component.run_async(
            [ChatMessage.from_user("What's the capital of France?")]
        )

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "gpt-4o" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert counter > 1
        assert "Paris" in responses

    @pytest.mark.asyncio
    async def test_streaming_callback_compatibility(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")

        async def async_callback(chunk: StreamingChunk):
            pass

        def sync_callback(chunk: StreamingChunk):
            pass

        with pytest.raises(ValueError, match="init callback must be async compatible"):
            gen = OpenAIChatGenerator(streaming_callback=sync_callback)
            await gen.run_async([])

        with pytest.raises(
            ValueError, match="runtime callback must be async compatible"
        ):
            gen = OpenAIChatGenerator(streaming_callback=async_callback)
            await gen.run_async([], streaming_callback=sync_callback)

        await gen.run_async([])

        with pytest.raises(ValueError, match="init callback cannot be a coroutine"):
            gen = OpenAIChatGenerator(streaming_callback=async_callback)
            gen.run([])

        with pytest.raises(ValueError, match="runtime callback cannot be a coroutine"):
            gen = OpenAIChatGenerator(streaming_callback=sync_callback)
            gen.run([], streaming_callback=async_callback)

        gen.run([])

    def test_convert_message_to_openai_format(self):
        message = ChatMessage.from_system("You are good assistant")
        assert _convert_message_to_openai_format(message) == {
            "role": "system",
            "content": "You are good assistant",
        }

        message = ChatMessage.from_user("I have a question")
        assert _convert_message_to_openai_format(message) == {
            "role": "user",
            "content": "I have a question",
        }

        message = ChatMessage.from_assistant(
            text="I have an answer", meta={"finish_reason": "stop"}
        )
        assert _convert_message_to_openai_format(message) == {
            "role": "assistant",
            "content": "I have an answer",
        }

        message = ChatMessage.from_user(
            text="Hello",
            media=[
                ByteStream(
                    data=b"test data", meta={"detail": "low"}, mime_type="image/png"
                )
            ],
        )
        assert _convert_message_to_openai_format(message) == {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,"
                        + b64encode(b"test data").decode("utf-8"),
                        "detail": "low",
                    },
                },
            ],
        }

        message = ChatMessage.from_assistant(
            tool_calls=[
                ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
            ]
        )
        assert _convert_message_to_openai_format(message) == {
            "content": [],
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "123",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }

        tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
        message = ChatMessage.from_tool(
            tool_result=tool_result,
            origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"}),
        )
        assert _convert_message_to_openai_format(message) == {
            "role": "tool",
            "content": tool_result,
            "tool_call_id": "123",
        }

    def test_convert_message_to_openai_invalid(self):
        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
        with pytest.raises(ValueError):
            _convert_message_to_openai_format(message)

        tool_call_null_id = ToolCall(
            id=None, tool_name="weather", arguments={"city": "Paris"}
        )
        message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
        with pytest.raises(ValueError):
            _convert_message_to_openai_format(message)

        message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
        with pytest.raises(ValueError):
            _convert_message_to_openai_format(message)

    def test_run_with_tools(self, tools):

        with patch(
            "openai.resources.chat.completions.Completions.create"
        ) as mock_chat_completion_create:
            completion = ChatCompletion(
                id="foo",
                model="gpt-4",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        logprobs=None,
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="123",
                                    type="function",
                                    function=Function(
                                        name="weather", arguments='{"city": "Paris"}'
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                created=int(datetime.now().timestamp()),
                usage={
                    "prompt_tokens": 57,
                    "completion_tokens": 40,
                    "total_tokens": 97,
                },
            )

            mock_chat_completion_create.return_value = completion

            component = OpenAIChatGenerator(
                api_key=Secret.from_token("test-api-key"), tools=tools
            )
            response = component.run(
                [ChatMessage.from_user("What's the weather like in Paris?")]
            )

        assert len(response["replies"]) == 1
        message = response["replies"][0]

        assert not message.texts
        assert not message.text

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    def test_run_with_tools_streaming(
        self, mock_chat_completion_chunk_with_tools, tools
    ):

        streaming_callback_called = False

        def streaming_callback(chunk: StreamingChunk) -> None:
            nonlocal streaming_callback_called
            streaming_callback_called = True

        component = OpenAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            streaming_callback=streaming_callback,
        )
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        response = component.run(chat_messages, tools=tools)

        # check we called the streaming callback
        assert streaming_callback_called

        # check that the component still returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

        message = response["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    def test_invalid_tool_call_json(self, tools, caplog):
        caplog.set_level(logging.WARNING)

        with patch(
            "openai.resources.chat.completions.Completions.create"
        ) as mock_create:
            mock_create.return_value = ChatCompletion(
                id="test",
                model="gpt-4o-mini",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="tool_calls",
                        index=0,
                        message=ChatCompletionMessage(
                            role="assistant",
                            tool_calls=[
                                ChatCompletionMessageToolCall(
                                    id="1",
                                    type="function",
                                    function=Function(
                                        name="weather", arguments='"invalid": "json"'
                                    ),
                                ),
                            ],
                        ),
                    )
                ],
                created=1234567890,
                usage={
                    "prompt_tokens": 50,
                    "completion_tokens": 30,
                    "total_tokens": 80,
                },
            )

            component = OpenAIChatGenerator(
                api_key=Secret.from_token("test-api-key"), tools=tools
            )
            response = component.run(
                [ChatMessage.from_user("What's the weather in Paris?")]
            )

        assert len(response["replies"]) == 1
        message = response["replies"][0]
        assert len(message.tool_calls) == 0
        assert (
            "OpenAI returned a malformed JSON string for tool call arguments"
            in caplog.text
        )

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):

        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = OpenAIChatGenerator(tools=tools)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert not message.texts
        assert not message.text
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_with_tools_async(self, tools):

        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = OpenAIChatGenerator(tools=tools)
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
