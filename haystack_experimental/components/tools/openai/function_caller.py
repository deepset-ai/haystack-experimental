# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List

from haystack import component
from haystack.dataclasses import ChatMessage


@component
class OpenAIFunctionCaller:
    """
    The OpenAIFunctionCaller expects a list of ChatMessages and if there is a tool call with a function name and arguments, it runs the function and returns the
    result as a ChatMessage from role = 'function'
    """

    def __init__(self, available_functions: Dict[str, Callable[...]):
        """
        Initialize the OpenAIFunctionCaller component.
        :param available_functions: A dictionary of available functions. This dictionary expects key value pairs of function name, and the function itself. For example, `{"weather_function": weather_function}`
        """
        self.available_functions = available_functions

    @component.output_types(
        function_replies=List[ChatMessage], assistant_replies=List[ChatMessage]
    )
    def run(self, messages: List[ChatMessage]):
        """
        Evaluates `messages` and invokes available functions if the messages contain tool_calls.
        :param messages: A list of messages generated from the `OpenAIChatGenerator`
        :returns: This component returns a list of messages in one of two outputs
            - `function_replies`: List of ChatMessages containing the result of a function invocation. This message comes from role = 'function'. If the function name was hallucinated or wrong, an assistant message explaining as such is returned
            - `assistant_replies`: List of ChatMessages containing a regular assistant reply. In this case, there were no tool_calls in the received messages
        """
        if messages[0].meta["finish_reason"] == "tool_calls":
            function_calls = json.loads(messages[0].content)
            for function_call in function_calls:
                function_name = function_call["function"]["name"]
                function_args = json.loads(function_call["function"]["arguments"])
                if function_name in self.available_functions:
                    function_to_call = self.available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    messages.append(
                        ChatMessage.from_function(
                            content=json.dumps(function_response), name=function_name
                        )
                    )
                else:
                    messages.append(
                        ChatMessage.from_assistant(
                            """I'm sorry, I tried to run a function that did not exist. 
                                                        Would you like me to correct it and try again?"""
                        )
                    )
            return {"function_replies": messages}
        return {"assistant_replies": messages}
