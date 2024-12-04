# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from base64 import b64encode
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.utils import (
    Secret,
    deserialize_callable,
    deserialize_secrets_inplace,
    serialize_callable,
)
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

from haystack_experimental.dataclasses import (
    ChatMessage,
    Tool,
    ToolCall,
    TextContent,
    ChatRole,
    MediaContent,
)
from haystack_experimental.dataclasses.streaming_chunk import (
    AsyncStreamingCallbackT,
    StreamingCallbackT,
    select_streaming_callback,
)
from haystack_experimental.dataclasses.tool import deserialize_tools_inplace

logger = logging.getLogger(__name__)


def _convert_message_to_openai_format(message: ChatMessage) -> Dict[str, Any]:
    """
    Convert a message to the format expected by OpenAI's Chat API.
    """
    openai_msg: Dict[str, Any] = {"role": message.role.value}
    if len(message) == 0:
        raise ValueError(
            "ChatMessage must contain at least one `TextContent`, "
            "`MediaContent`, `ToolCall`, or `ToolCallResult`."
        )
    if len(message) == 1 and isinstance(message.content[0], TextContent):
        openai_msg["content"] = message.content[0].text
    elif message.tool_call_result:
        # Tool call results should only be included for ChatRole.TOOL messages
        # and should not include any other content
        if message.role != ChatRole.TOOL:
            raise ValueError(
                "Tool call results should only be included for tool messages."
            )
        if len(message) > 1:
            raise ValueError(
                "Tool call results should not be included with other content."
            )
        if message.tool_call_result.origin.id is None:
            raise ValueError(
                "`ToolCall` must have a non-null `id` attribute to be used with OpenAI."
            )
        openai_msg["content"] = message.tool_call_result.result
        openai_msg["tool_call_id"] = message.tool_call_result.origin.id
    else:
        openai_msg["content"] = []
        for item in message.content:
            if isinstance(item, TextContent):
                openai_msg["content"].append({"type": "text", "text": item.text})
            elif isinstance(item, MediaContent):
                match item.media.type:
                    case "image":
                        base64_data = b64encode(item.media.data).decode("utf-8")
                        url = f"data:{item.media.mime_type};base64,{base64_data}"
                        openai_msg["content"].append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": url,
                                    "detail": item.media.meta.get("detail", "auto"),
                                },
                            }
                        )
                    case _:
                        raise ValueError(
                            f"Unsupported media type '{item.media.mime_type}' for OpenAI completions."
                        )
            elif isinstance(item, ToolCall):
                if message.role != ChatRole.ASSISTANT:
                    raise ValueError(
                        "Tool calls should only be included for assistant messages."
                    )
                if item.id is None:
                    raise ValueError(
                        "`ToolCall` must have a non-null `id` attribute to be used with OpenAI."
                    )
                openai_msg.setdefault("tool_calls", []).append(
                    {
                        "id": item.id,
                        "type": "function",
                        "function": {
                            "name": item.tool_name,
                            "arguments": json.dumps(item.arguments, ensure_ascii=False),
                        },
                    }
                )
            else:
                raise ValueError(
                    f"Unsupported content type '{type(item).__name__}' for OpenAI completions."
                )

    if message.name:
        openai_msg["name"] = message.name

    return openai_msg


@component
class OpenAIChatGenerator:
    """
    Completes chats using OpenAI's large language models (LLMs).

    It works with the gpt-4 and gpt-3.5-turbo models and supports streaming responses
    from OpenAI API. It uses [ChatMessage](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)
    format in input and output.

    You can customize how the text is generated by passing parameters to the
    OpenAI API. Use the `**generation_kwargs` argument when you initialize
    the component or when you run it. Any parameter that works with
    `openai.ChatCompletion.create` will work here too.

    For details on OpenAI API parameters, see
    [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat).

    ### Usage example

    ```python
    from haystack_experimental.components.generators.chat import OpenAIChatGenerator
    from haystack_experimental.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = OpenAIChatGenerator()
    response = client.run(messages)
    print(response)
    ```
    Output:
    ```
    {'replies': [
        ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>,
                    _content=[TextContent(text='Natural Language Processing (NLP) is a field of artificial ...')],
                    _meta={'model': 'gpt-4o-mini', 'index': 0, 'finish_reason': 'stop',
                        'usage': {'completion_tokens': 71, 'prompt_tokens': 13, 'total_tokens': 84}}
                    )
                ]
    }
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "gpt-4o-mini",
        streaming_callback: Optional[
            Union[StreamingCallbackT, AsyncStreamingCallbackT]
        ] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: bool = False,
    ):
        """
        Creates an instance of OpenAIChatGenerator.

        :param api_key:
            The OpenAI API key.
        :param model:
            The name of the model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
            as an argument. Must be a coroutine if the component is used in an async pipeline.
        :param api_base_url:
            An optional base URL.
        :param organization:
            Your organization ID. See [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are sent directly to the OpenAI endpoint.
            See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. For example, 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
                values are the bias to add to that token.
        :param timeout:
            Timeout for OpenAI client calls. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param tools:
            A list of tools for which the model can prepare calls.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
        """
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self.tools = tools
        self.tools_strict = tools_strict

        self._validate_tools(tools)

        if timeout is None:
            timeout = float(os.environ.get("OPENAI_TIMEOUT", 30.0))
        if max_retries is None:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", 5))

        self.client = OpenAI(
            api_key=api_key.resolve_value(),
            organization=organization,
            base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key.resolve_value(),
            organization=organization,
            base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = (
            serialize_callable(self.streaming_callback)
            if self.streaming_callback
            else None
        )
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            tools=[tool.to_dict() for tool in self.tools] if self.tools else None,
            tools_strict=self.tools_strict,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(
                serialized_callback_handler
            )

        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(  # noqa: PLR0913
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[
            Union[StreamingCallbackT, AsyncStreamingCallbackT]
        ] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: Optional[bool] = None,
    ):
        """
        Invokes chat completion based on the provided messages and generation parameters.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            Cannot be a coroutine.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A list containing the generated responses as ChatMessage instances.
        """
        # validate and select the streaming callback
        streaming_callback = select_streaming_callback(
            self.streaming_callback, streaming_callback, requires_async=False
        )  # type: ignore

        if len(messages) == 0:
            return {"replies": []}

        api_args = self._prepare_api_call(
            messages, streaming_callback, generation_kwargs, tools, tools_strict
        )
        chat_completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = (
            self.client.chat.completions.create(**api_args)
        )

        is_streaming = isinstance(chat_completion, Stream)
        assert is_streaming or streaming_callback is None

        if is_streaming:
            completions = self._handle_stream_response(
                chat_completion,  # type: ignore
                streaming_callback,  # type: ignore
            )
        else:
            assert isinstance(
                chat_completion, ChatCompletion
            ), "Unexpected response type for non-streaming request."
            completions = [
                self._convert_chat_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            self._check_finish_reason(message.meta)

        return {"replies": completions}

    @component.output_types(replies=List[ChatMessage])
    async def run_async(  # noqa: PLR0913
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[
            Union[StreamingCallbackT, AsyncStreamingCallbackT]
        ] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: Optional[bool] = None,
    ):
        """
        Invokes chat completion based on the provided messages and generation parameters.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            Must be a coroutine.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A list containing the generated responses as ChatMessage instances.
        """
        # validate and select the streaming callback
        streaming_callback = select_streaming_callback(self.streaming_callback, streaming_callback, requires_async=True)  # type: ignore

        if len(messages) == 0:
            return {"replies": []}

        api_args = self._prepare_api_call(
            messages, streaming_callback, generation_kwargs, tools, tools_strict
        )
        chat_completion: Union[AsyncStream[ChatCompletionChunk], ChatCompletion] = (
            await self.async_client.chat.completions.create(**api_args)
        )

        is_streaming = isinstance(chat_completion, AsyncStream)
        assert is_streaming or streaming_callback is None

        if is_streaming:
            completions = await self._handle_async_stream_response(
                chat_completion,  # type: ignore
                streaming_callback,  # type: ignore
            )
        else:
            assert isinstance(
                chat_completion, ChatCompletion
            ), "Unexpected response type for non-streaming request."
            completions = [
                self._convert_chat_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            self._check_finish_reason(message.meta)

        return {"replies": completions}

    def _validate_tools(self, tools: Optional[List[Tool]]):
        if tools is None:
            return

        tool_names = [tool.name for tool in tools]
        duplicate_tool_names = {
            name for name in tool_names if tool_names.count(name) > 1
        }
        if duplicate_tool_names:
            raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")

    def _prepare_api_call(  # noqa: PLR0913
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[
            Union[StreamingCallbackT, AsyncStreamingCallbackT]
        ],
        generation_kwargs: Optional[Dict[str, Any]],
        tools: Optional[List[Tool]],
        tools_strict: Optional[bool],
    ) -> Dict[str, Any]:
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [
            _convert_message_to_openai_format(message) for message in messages
        ]

        tools = tools or self.tools
        tools_strict = tools_strict if tools_strict is not None else self.tools_strict
        self._validate_tools(tools)

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {**t.tool_spec, "strict": tools_strict},
                }
                for t in tools
            ]

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)
        if is_streaming and num_responses > 1:
            raise ValueError("Cannot stream multiple responses, please set n=1.")

        return {
            "model": self.model,
            "messages": openai_formatted_messages,  # type: ignore[arg-type] # openai expects list of specific message types
            "stream": streaming_callback is not None,
            "tools": openai_tools,  # type: ignore[arg-type]
            "n": num_responses,
            **generation_kwargs,
        }

    def _handle_stream_response(
        self,
        chat_completion: Stream,
        callback: StreamingCallbackT,
    ) -> List[ChatMessage]:
        chunks: List[StreamingChunk] = []
        chunk = None

        for chunk in chat_completion:  # pylint: disable=not-an-iterable
            assert (
                len(chunk.choices) == 1
            ), "Streaming responses should have only one choice."
            chunk_delta: StreamingChunk = (
                self._convert_chat_completion_chunk_to_streaming_chunk(chunk)
            )
            chunks.append(chunk_delta)

            callback(chunk_delta)

        return [self._convert_streaming_chunks_to_chat_message(chunk, chunks)]

    async def _handle_async_stream_response(
        self,
        chat_completion: AsyncStream,
        callback: AsyncStreamingCallbackT,
    ) -> List[ChatMessage]:
        chunks: List[StreamingChunk] = []
        chunk = None

        async for chunk in chat_completion:  # pylint: disable=not-an-iterable
            assert (
                len(chunk.choices) == 1
            ), "Streaming responses should have only one choice."
            chunk_delta: StreamingChunk = (
                self._convert_chat_completion_chunk_to_streaming_chunk(chunk)
            )
            chunks.append(chunk_delta)

            await callback(chunk_delta)

        return [self._convert_streaming_chunks_to_chat_message(chunk, chunks)]

    def _check_finish_reason(self, meta: Dict[str, Any]) -> None:
        if meta["finish_reason"] == "length":
            logger.warning(
                "The completion for index {index} has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                index=meta["index"],
                finish_reason=meta["finish_reason"],
            )
        if meta["finish_reason"] == "content_filter":
            logger.warning(
                "The completion for index {index} has been truncated due to the content filter.",
                index=meta["index"],
                finish_reason=meta["finish_reason"],
            )

    def _convert_streaming_chunks_to_chat_message(
        self, chunk: Any, chunks: List[StreamingChunk]
    ) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.

        :param chunk: The last chunk returned by the OpenAI API.
        :param chunks: The list of all `StreamingChunk` objects.
        """

        text = "".join([chunk.content for chunk in chunks])
        tool_calls = []

        # if it's a tool call , we need to build the payload dict from all the chunks
        if bool(chunks[0].meta.get("tool_calls")):
            tools_len = len(chunks[0].meta.get("tool_calls", []))

            payloads = [{"arguments": "", "name": ""} for _ in range(tools_len)]
            for chunk_payload in chunks:
                deltas = chunk_payload.meta.get("tool_calls") or []

                # deltas is a list of ChoiceDeltaToolCall or ChoiceDeltaFunctionCall
                for i, delta in enumerate(deltas):
                    payloads[i]["id"] = delta.id or payloads[i].get("id", "")
                    if delta.function:
                        payloads[i]["name"] += delta.function.name or ""
                        payloads[i]["arguments"] += delta.function.arguments or ""

            for payload in payloads:
                arguments_str = payload["arguments"]
                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append(
                        ToolCall(
                            id=payload["id"],
                            tool_name=payload["name"],
                            arguments=arguments,
                        )
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "OpenAI returned a malformed JSON string for tool call arguments. This tool call "
                        "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                        "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                        _id=payload["id"],
                        _name=payload["name"],
                        _arguments=arguments_str,
                    )

        meta = {
            "model": chunk.model,
            "index": 0,
            "finish_reason": chunk.choices[0].finish_reason,
            "usage": {},  # we don't have usage data for streaming responses
        }

        return ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta)

    def _convert_chat_completion_to_chat_message(
        self, completion: ChatCompletion, choice: Choice
    ) -> ChatMessage:
        """
        Converts the non-streaming response from the OpenAI API to a ChatMessage.

        :param completion: The completion returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The ChatMessage.
        """
        message: ChatCompletionMessage = choice.message
        text = message.content
        tool_calls = []
        if openai_tool_calls := message.tool_calls:
            for openai_tc in openai_tool_calls:
                arguments_str = openai_tc.function.arguments
                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append(
                        ToolCall(
                            id=openai_tc.id,
                            tool_name=openai_tc.function.name,
                            arguments=arguments,
                        )
                    )
                except json.JSONDecodeError:
                    logger.warning(
                        "OpenAI returned a malformed JSON string for tool call arguments. This tool call "
                        "will be skipped. To always generate a valid JSON, set `tools_strict` to `True`. "
                        "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                        _id=openai_tc.id,
                        _name=openai_tc.function.name,
                        _arguments=arguments_str,
                    )

        chat_message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls)
        chat_message._meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage or {}),
            }
        )
        return chat_message

    def _convert_chat_completion_chunk_to_streaming_chunk(
        self, chunk: ChatCompletionChunk
    ) -> StreamingChunk:
        """
        Converts the streaming response chunk from the OpenAI API to a StreamingChunk.

        :param chunk: The chunk returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The StreamingChunk.
        """
        # we stream the content of the chunk if it's not a tool or function call
        choice: ChunkChoice = chunk.choices[0]
        content = choice.delta.content or ""
        chunk_message = StreamingChunk(content)
        # but save the tool calls and function call in the meta if they are present
        # and then connect the chunks in the _convert_streaming_chunks_to_chat_message method
        chunk_message.meta.update(
            {
                "model": chunk.model,
                "index": choice.index,
                "tool_calls": choice.delta.tool_calls,
                "finish_reason": choice.finish_reason,
            }
        )
        return chunk_message
