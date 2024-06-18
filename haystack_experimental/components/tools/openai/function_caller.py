# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Callable, Dict, List

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.utils import deserialize_callable, serialize_callable

_FUNCTION_NAME_FAILURE = (
    "I'm sorry, I tried to run a function that did not exist. Would you like me to correct it and try again?"
)
_FUNCTION_RUN_FAILURE = "Seems there was an error while running the function: {error}"


@component
class OpenAIFunctionCaller:
    """
    OpenAIFunctionCaller processes a list of chat messages and call Python functions when needed.

    The OpenAIFunctionCaller expects a list of ChatMessages and if there is a tool call with a function name and
    arguments, it runs the function and returns the result as a ChatMessage from role = 'function'
    """

    def __init__(self, available_functions: Dict[str, Callable]):
        """
        Initialize the OpenAIFunctionCaller component.

        :param available_functions:
            A dictionary of available functions. This dictionary expects key value pairs of function name,
            and the function itself. For example, `{"weather_function": weather_function}`
        """
        self.available_functions = available_functions

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        available_function_paths = {}
        for name, function in self.available_functions.items():
            available_function_paths[name] = serialize_callable(function)
        serialization_dict = default_to_dict(self, available_functions=available_function_paths)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIFunctionCaller":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        available_function_paths = data.get("init_parameters", {}).get("available_functions")
        available_functions = {}
        for name, path in available_function_paths.items():
            available_functions[name] = deserialize_callable(path)
        data["init_parameters"]["available_functions"] = available_functions
        return default_from_dict(cls, data)

    @component.output_types(function_replies=List[ChatMessage], assistant_replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        """
        Evaluates `messages` and invokes available functions if the messages contain tool_calls.

        :param messages: A list of messages generated from the `OpenAIChatGenerator`
        :returns: This component returns a list of messages in one of two outputs
            - `function_replies`: List of ChatMessages containing the result of a function invocation.
                This message comes from role = 'function'. If the function name was hallucinated or wrong,
                an assistant message explaining as such is returned
            - `assistant_replies`: List of ChatMessages containing a regular assistant reply. In this case,
                there were no tool_calls in the received messages
        """
        if messages[0].meta["finish_reason"] == "tool_calls":
            function_calls = json.loads(messages[0].content)
            for function_call in function_calls:
                function_name = function_call["function"]["name"]
                function_args = json.loads(function_call["function"]["arguments"])
                if function_name in self.available_functions:
                    function_to_call = self.available_functions[function_name]
                    try:
                        function_response = function_to_call(**function_args)
                        messages.append(
                            ChatMessage.from_function(
                                content=json.dumps(function_response),
                                name=function_name,
                            )
                        )
                    # pylint: disable=broad-exception-caught
                    except Exception as e:
                        messages.append(ChatMessage.from_assistant(_FUNCTION_RUN_FAILURE.format(error=e)))
                else:
                    messages.append(ChatMessage.from_assistant(_FUNCTION_NAME_FAILURE))
            return {"function_replies": messages}
        return {"assistant_replies": messages}
