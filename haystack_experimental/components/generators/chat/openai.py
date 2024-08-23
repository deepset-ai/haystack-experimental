# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Callable, Dict, List, Optional, Union

from haystack import component, default_from_dict, logging
from haystack.components.generators.chat.openai import OpenAIChatGenerator as OpenAIChatGeneratorBase
from haystack.dataclasses import StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

from haystack_experimental.dataclasses import ChatMessage, ToolCall
from haystack_experimental.dataclasses.tool import Tool, deserialize_tools_inplace

logger = logging.getLogger(__name__)


def _convert_message_to_openai_format(message: ChatMessage) -> Dict[str, Any]:
    """
    Convert a message to the format expected by OpenAI's Chat API.
    """
    text_contents = message.texts
    tool_calls = message.tool_calls
    tool_call_results = message.tool_call_results

    if not text_contents and not tool_calls and not tool_call_results:
        raise ValueError("A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`.")
    elif len(text_contents) + len(tool_call_results) > 1:
        raise ValueError("A `ChatMessage` can only contain one `TextContent` or one `ToolCallResult`.")

    openai_msg: Dict[str, Any] = {"role": message._role.value}

    if tool_call_results:
        result = tool_call_results[0]
        if result.origin.id is None:
            raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
        openai_msg["content"] = result.result
        openai_msg["tool_call_id"] = result.origin.id
        # OpenAI does not provide a way to communicate errors in tool invocations, so we ignore the error field
        return openai_msg

    if text_contents:
        openai_msg["content"] = text_contents[0]
    if tool_calls:
        openai_tool_calls = []
        for tc in tool_calls:
            if tc.id is None:
                raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
            openai_tool_calls.append(
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments)},
                }
            )
        openai_msg["tool_calls"] = openai_tool_calls
    return openai_msg


@component
class OpenAIChatGenerator(OpenAIChatGeneratorBase):
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
                    _meta={'model': 'gpt-3.5-turbo-0125', 'index': 0, 'finish_reason': 'stop',
                        'usage': {'completion_tokens': 71, 'prompt_tokens': 13, 'total_tokens': 84}}
                    )
                ]
    }
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "gpt-3.5-turbo",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: bool = False,
    ):
        """
        Creates an instance of OpenAIChatGenerator. Unless specified otherwise in `model`, uses OpenAI's GPT-3.5.

        Before initializing the component, you can set the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES'
        environment variables to override the `timeout` and `max_retries` parameters respectively
        in the OpenAI client.

        :param api_key: The OpenAI API key.
            You can set it with an environment variable `OPENAI_API_KEY`, or pass with this parameter
            during initialization.
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
            as an argument.
        :param api_base_url: An optional base URL.
        :param organization: Your organization ID, defaults to `None`. See
        [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
        :param generation_kwargs: Other parameters to use for the model. These parameters are sent directly to
            the OpenAI endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for
            more details.
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
        if tools:
            tool_names = [tool.name for tool in tools]
            duplicate_tool_names = {name for name in tool_names if tool_names.count(name) > 1}
            if duplicate_tool_names:
                raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")
        self.tools = tools
        self.tools_strict = tools_strict

        super(OpenAIChatGenerator, self).__init__(
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=organization,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        serialized = super(OpenAIChatGenerator, self).to_dict()
        serialized["init_parameters"]["tools"] = [tool.to_dict() for tool in self.tools] if self.tools else None
        serialized["init_parameters"]["tools_strict"] = self.tools_strict
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(  # noqa: PLR0913
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
        tools_strict: Optional[bool] = None,
    ):
        """
        Invokes chat completion based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
                                  override the parameters passed during component initialization.
                                  For details on OpenAI API parameters, see
                                  [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
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

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # check if streaming_callback is passed
        streaming_callback = streaming_callback or self.streaming_callback

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [_convert_message_to_openai_format(message) for message in messages]

        tools = tools or self.tools
        if tools:
            tool_names = [tool.name for tool in tools]
            duplicate_tool_names = {name for name in tool_names if tool_names.count(name) > 1}
            if duplicate_tool_names:
                raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")

        tools_strict = tools_strict if tools_strict is not None else self.tools_strict

        openai_tools = None
        if tools:
            openai_tools = [{"type": "function", "function": {**t.tool_spec, "strict": tools_strict}} for t in tools]

        chat_completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            model=self.model,
            messages=openai_formatted_messages,  # type: ignore[arg-type] # openai expects list of specific message types
            stream=streaming_callback is not None,
            tools=openai_tools,  # type: ignore[arg-type]
            **generation_kwargs,
        )

        completions: List[ChatMessage] = []
        # if streaming is enabled, the completion is a Stream of ChatCompletionChunk
        if isinstance(chat_completion, Stream):
            num_responses = generation_kwargs.pop("n", 1)
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")
            chunks: List[StreamingChunk] = []
            chunk = None

            # pylint: disable=not-an-iterable
            for chunk in chat_completion:
                if chunk.choices and streaming_callback:
                    chunk_delta: StreamingChunk = self._convert_chat_completion_chunk_to_streaming_chunk(chunk)
                    chunks.append(chunk_delta)
                    streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
            completions = [self._convert_streaming_chunks_to_chat_message(chunk, chunks)]
        # if streaming is disabled, the completion is a ChatCompletion
        elif isinstance(chat_completion, ChatCompletion):
            completions = [
                self._convert_chat_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            self._check_finish_reason(message)

        return {"replies": completions}

    def _convert_streaming_chunks_to_chat_message(self, chunk: Any, chunks: List[StreamingChunk]) -> ChatMessage:
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
                    tool_calls.append(ToolCall(id=payload["id"], tool_name=payload["name"], arguments=arguments))
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

    def _convert_chat_completion_to_chat_message(self, completion: ChatCompletion, choice: Choice) -> ChatMessage:
        """
        Converts the non-streaming response from the OpenAI API to a ChatMessage.

        :param completion: The completion returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The ChatMessage.
        """
        message: ChatCompletionMessage = choice.message
        text = message.content or ""
        tool_calls = []
        if openai_tool_calls := message.tool_calls:
            for openai_tc in openai_tool_calls:
                arguments_str = openai_tc.function.arguments
                try:
                    arguments = json.loads(arguments_str)
                    tool_calls.append(ToolCall(id=openai_tc.id, tool_name=openai_tc.function.name, arguments=arguments))
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

    def _convert_chat_completion_chunk_to_streaming_chunk(self, chunk: ChatCompletionChunk) -> StreamingChunk:
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
