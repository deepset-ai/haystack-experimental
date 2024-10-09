from haystack import component
from haystack.components.generators.openai import OpenAIGenerator as OpenAIGeneratorBase
from haystack.components.generators.openai_utils import (
    _convert_message_to_openai_format,
)
from haystack.dataclasses import ChatMessage, StreamingChunk
from openai import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from typing import List, Any, Dict, Optional, Callable, Union
from pydantic import BaseModel


@component
class OpenAIGenerator(OpenAIGeneratorBase):
    """
    Generates text using OpenAI's large language models (LLMs).

    This is an experimental extension to the `OpenAIGenerator` which indcludes the structured outputs capabilities of OpenAI.
    To learn more about this feature which is still in Beta, visit the OpenAI docs: https://platform.openai.com/docs/guides/structured-outputs

    The structured output feature works with gpt-4o-mini and later. For a list of compatible models, refer to the OpenAI documentation.
    This component expects a Pydantic BaseModel provided as the `"response_format"` in `generation_kwargs`.

    ### Usage example
    To use the `OpenAIGenerator` with the structured output feature, here's an example:
    ```
    from haystack_experimental.components.generators import OpenAIGenerator
    from pydantic import BaseModel

    class Question(BaseModel):
        question: str
        answer: Optional[str] = None

    class Questions(BaseModel):
        questions: list[Question]

    OpenAIGenerator(model="gpt-4o-mini", generation_kwargs={"response_format": Questions})
    ```
    """

    def __init__(self, **kwargs):
        """
        Creates an instance of OpenAIGenerator. Unless specified otherwise in `model`, uses OpenAI's GPT-3.5.

        By setting the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES' you can change the timeout and max_retries parameters
        in the OpenAI client.

        :param api_key: The OpenAI API key to connect to OpenAI.
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url: An optional base URL.
        :param organization: The Organization ID, defaults to `None`.
        :param system_prompt: The system prompt to use for text generation. If not provided, the system prompt is
        omitted, and the default system prompt of the model is used.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the OpenAI endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for
            more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So, 0.1 means only the tokens
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
            Timeout for OpenAI Client calls, if not set it is inferred from the `OPENAI_TIMEOUT` environment variable
            or set to 30.
        :param max_retries:
            Maximum retries to establish contact with OpenAI if it returns an internal error, if not set it is inferred
            from the `OPENAI_MAX_RETRIES` environment variable or set to 5.

        """
        super().__init__(**kwargs)

    @component.output_types(
        replies=List[str], meta=List[Dict[str, Any]], structured_reply=BaseModel
    )
    def run(
        self,
        prompt: str,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt:
            The string prompt to use for text generation.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. To use the structured output feature, provide a Pydantic model
            in "response_format". These parameters will potentially override the parameters
            passed in the `__init__` method. For more details on the parameters supported by the OpenAI API, refer to
            the OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :returns:
            A list of strings containing the generated responses and a list of dictionaries containing the metadata
        for each response.
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        if "response_format" in generation_kwargs.keys():
            message = ChatMessage.from_user(prompt)
            if self.system_prompt:
                messages = [ChatMessage.from_system(self.system_prompt), message]
            else:
                messages = [message]

            streaming_callback = streaming_callback or self.streaming_callback
            openai_formatted_messages = [
                _convert_message_to_openai_format(message) for message in messages
            ]
            completion: Union[
                Stream[ChatCompletionChunk], ChatCompletion
            ] = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=openai_formatted_messages,
                **generation_kwargs
            )
            completions = [
                self._build_structured_message(completion, choice)
                for choice in completion.choices
            ]
            for response in completions:
                self._check_finish_reason(response)

            return {
                "replies": [message.content for message in completions],
                "meta": [message.meta for message in completions],
                "structured_reply": completions[0].content,
            }
        else:
            return super().run(prompt, streaming_callback, generation_kwargs)

    def _build_structured_message(self, completion: Any, choice: Any) -> ChatMessage:
        chat_message = ChatMessage.from_assistant(choice.message.parsed or "")
        chat_message.meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage),
            }
        )
        return chat_message
