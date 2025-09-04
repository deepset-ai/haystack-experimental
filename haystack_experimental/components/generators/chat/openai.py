# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from typing import Any, Optional, Union

from haystack import component
from haystack.components.generators.chat.openai import OpenAIChatGenerator as BaseOpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool, Toolset

from haystack_experimental.utils.hallucination_risk_calculator.dataclasses import HallucinationScoreConfig
from haystack_experimental.utils.hallucination_risk_calculator.openai_planner import calculate_hallucination_metrics


@component
class OpenAIChatGenerator(BaseOpenAIChatGenerator):
    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        tools: Optional[Union[list[Tool], Toolset]] = None,
        tools_strict: Optional[bool] = None,
        hallucination_score_config: Optional[HallucinationScoreConfig] = None,
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
        :param hallucination_score_config:
            If provided, the generator will evaluate the hallucination risk of its responses using
            the OpenAIPlanner and annotate each response with hallucination metrics.
            This involves generating multiple samples and analyzing their consistency, which may increase
            latency and cost. Use this option when you need to assess the reliability of the generated content
            in scenarios where accuracy is critical.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances. If hallucination
              scoring is enabled, each message will include additional metadata:
                - `hallucination_decision`: "ANSWER" if the model decided to answer, "REFUSE" if it abstained.
                - `hallucination_risk`: The EDFL hallucination risk bound.
                - `hallucination_rationale`: The rationale behind the hallucination decision.
        """
        if len(messages) == 0:
            return {"replies": []}

        # Call parent implementation
        result = super(OpenAIChatGenerator, self).run(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        completions = result["replies"]

        # Add hallucination scoring if configured
        if hallucination_score_config:
            hallucination_meta = calculate_hallucination_metrics(
                prompt=messages[-1].text,
                hallucination_score_config=hallucination_score_config,
                chat_generator=self
            )
            completions = [replace(m, _meta={**m.meta, **hallucination_meta}) for m in completions]

        return {"replies": completions}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        *,
        tools: Optional[Union[list[Tool], Toolset]] = None,
        tools_strict: Optional[bool] = None,
        hallucination_score_config: Optional[HallucinationScoreConfig] = None,
    ):
        """
        Asynchronously invokes chat completion based on the provided messages and generation parameters.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

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
            A list of tools or a Toolset for which the model can prepare calls. If set, it will override the
            `tools` parameter set during component initialization. This parameter can accept either a list of
            `Tool` objects or a `Toolset` instance.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.
        :param hallucination_score_config:
            If provided, the generator will evaluate the hallucination risk of its responses using
            the OpenAIPlanner and annotate each response with hallucination metrics.
            This involves generating multiple samples and analyzing their consistency, which may increase
            latency and cost. Use this option when you need to assess the reliability of the generated content
            in scenarios where accuracy is critical.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances. If hallucination
              scoring is enabled, each message will include additional metadata:
                - `hallucination_decision`: "ANSWER" if the model decided to answer, "REFUSE" if it abstained.
                - `hallucination_risk`: The EDFL hallucination risk bound.
                - `hallucination_rationale`: The rationale behind the hallucination decision.
        """
        if len(messages) == 0:
            return {"replies": []}

        # Call parent implementation
        result = await super(OpenAIChatGenerator, self).run_async(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        completions = result["replies"]

        # Add hallucination scoring if configured
        if hallucination_score_config:
            hallucination_meta = calculate_hallucination_metrics(
                prompt=messages[-1].text,
                hallucination_score_config=hallucination_score_config,
                chat_generator=self
            )
            completions = [replace(m, _meta={**m.meta, **hallucination_meta}) for m in completions]

        return {"replies": completions}
