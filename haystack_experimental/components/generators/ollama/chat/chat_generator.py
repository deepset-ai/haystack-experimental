# from haystack.dataclasses import ChatMessage, StreamingChunk
from typing import Any, Callable, Dict, List, Optional, Type, Union

from haystack import component, default_from_dict
from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

from haystack_experimental.dataclasses import ChatMessage, ToolCall
from haystack_experimental.dataclasses.tool import Tool, deserialize_tools_inplace

with LazyImport("Run 'pip install ollama-haystack'") as ollama_integration_import:
    # pylint: disable=import-error
    from haystack_integrations.components.generators.ollama import OllamaChatGenerator as OllamaChatGeneratorBase

base_class: Union[Type[object], Type["OllamaChatGeneratorBase"]] = object
if ollama_integration_import.is_successful():
    base_class = OllamaChatGeneratorBase

print(base_class)


@component()
class OllamaChatGenerator(base_class):
    """
    Supports models running on Ollama.

    Find the full list of supported models [here](https://ollama.ai/library).

    Usage example:
    ```python
    from haystack_experimental.components.generators.ollama import OllamaChatGenerator
    from haystack_experimental.dataclasses import ChatMessage

    generator = OllamaChatGenerator(model="zephyr",
                                url = "http://localhost:11434",
                                generation_kwargs={
                                "num_predict": 100,
                                "temperature": 0.9,
                                })

    messages = [ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
    ChatMessage.from_user("What's Natural Language Processing?")]

    print(generator.run(messages=messages))
    ```
    """

    def __init__(
        self,
        model: str = "orca-mini",
        url: str = "http://localhost:11434",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Creates an instance of OllamaChatGenerator.

        :param model:
            The name of the model to use. The model should be available in the running Ollama instance.
        :param url:
            The URL of a running Ollama instance.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, and others. See the available arguments in
            [Ollama docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param timeout:
            The number of seconds before throwing a timeout error from the Ollama API.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param tools:
            A list of tools for which the model can prepare calls.
            Not all models support tools. For a list of models compatible with tools, see the
            [models page](https://ollama.com/search?c=tools).
        """
        ollama_integration_import.check()

        if tools:
            tool_names = [tool.name for tool in tools]
            duplicate_tool_names = {name for name in tool_names if tool_names.count(name) > 1}
            if duplicate_tool_names:
                raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")
        self.tools = tools

        super(OllamaChatGenerator, self).__init__(
            model=model,
            url=url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            streaming_callback=streaming_callback,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        serialized = super(OllamaChatGenerator, self).to_dict()
        serialized["init_parameters"]["tools"] = [tool.to_dict() for tool in self.tools] if self.tools else None
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OllamaChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)

    # TODO: rework
    def _message_to_dict(self, message: ChatMessage) -> Dict[str, str]:
        return {"role": message.role.value, "content": message.text or ""}

    # TODO: rework
    def _build_message_from_ollama_response(self, ollama_response: Dict[str, Any]) -> ChatMessage:
        """
        Converts the non-streaming response from the Ollama API to a ChatMessage.
        """
        message = ChatMessage.from_assistant(text=ollama_response["message"]["content"])
        message.meta.update({key: value for key, value in ollama_response.items() if key != "message"})
        return message

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Runs an Ollama Model on a given chat history.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param generation_kwargs:
            Optional arguments to pass to the Ollama generation endpoint, such as temperature,
            top_p, etc. See the
            [Ollama docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values).
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :returns: A dictionary with the following keys:
            - `replies`: The responses from the model
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        stream = self.streaming_callback is not None

        tools = tools or self.tools
        if tools:
            if stream:
                raise ValueError("Ollama does not support tools and streaming at the same time. Please choose one.")
            tool_names = [tool.name for tool in tools]
            duplicate_tool_names = {name for name in tool_names if tool_names.count(name) > 1}
            if duplicate_tool_names:
                raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")

        # ollama_tools = None
        # if tools:
        #     ollama_tools = [{"type": "function", "function": {**t.tool_spec}} for t in tools]

        messages = [self._message_to_dict(message) for message in messages]
        response = self._client.chat(model=self.model, messages=messages, stream=stream, options=generation_kwargs)

        if stream:
            chunks: List[StreamingChunk] = self._handle_streaming_response(response)
            return self._convert_to_streaming_response(chunks)

        return {"replies": [self._build_message_from_ollama_response(response)]}
