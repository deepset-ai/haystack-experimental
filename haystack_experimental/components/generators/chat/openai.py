# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Any, Optional, Union

from openai import Stream
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)

from haystack import component
from haystack.components.generators.chat.openai import (
    OpenAIChatGenerator,
    _convert_chat_completion_to_chat_message,
    _check_finish_reason,
)
from haystack.dataclasses import (
    ChatMessage,
    StreamingCallbackT,
    select_streaming_callback,
)
from haystack.tools import Tool, Toolset

from haystack_experimental.utils.hallucination_risk_calculator import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
)


@component
class OpenAIChatGenerator(OpenAIChatGenerator):
    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        tools: Optional[Union[list[Tool], Toolset]] = None,
        tools_strict: Optional[bool] = None,
        hallucination_score: bool = False,
    ):
        """
        Invokes chat completion based on the provided messages and generation parameters.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. If set, it will override the
            `tools` parameter set during component initialization. This parameter can accept either a list of
            `Tool` objects or a `Toolset` instance.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.
        :param hallucination_score:
            If set to `True`, the generator will evaluate the hallucination risk of its responses using
            the OpenAIPlanner and annotate each response with hallucination metrics.
            This involves generating multiple samples and analyzing their consistency, which may increase
            latency and cost. Use this option when you need to assess the reliability of the generated content
            in scenarios where accuracy is critical.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if len(messages) == 0:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        # Calculate the hallucination score pre-emptively on the last user message
        hallucination_meta = {}
        if hallucination_score:
            backend = OpenAIBackend(model="gpt-4o-mini", api_key=self.api_key.resolve_value())
            item = OpenAIItem(prompt=messages[-1].text, n_samples=7, m=6, skeleton_policy="closed_book")
            planner = OpenAIPlanner(backend, temperature=0.3)
            metrics = planner.run(
                [item], h_star=0.05, isr_threshold=1.0, margin_extra_bits=0.2, B_clip=12.0, clip_mode="one-sided"
            )
            hallucination_meta = {
                "hallucination_decision": "ANSWER" if metrics[0].decision_answer else "REFUSE",
                "hallucination_risk": metrics[0].roh_bound,
                "hallucination_rationale": metrics[0].rationale,
            }

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        chat_completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            **api_args
        )

        if streaming_callback is not None:
            completions = self._handle_stream_response(
                # we cannot check isinstance(chat_completion, Stream) because some observability tools wrap Stream
                # and return a different type. See https://github.com/deepset-ai/haystack/issues/9014.
                chat_completion,  # type: ignore
                streaming_callback,
            )

        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."
            completions = [
                _convert_chat_completion_to_chat_message(chat_completion, choice) for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            _check_finish_reason(message.meta)

        if hallucination_meta:
            completions = [replace(message, meta={**message.meta, **hallucination_meta}) for message in completions]

        return {"replies": completions}
