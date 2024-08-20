# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List

from haystack import component, default_from_dict, default_to_dict, logging

from haystack_experimental.dataclasses import ChatMessage, ChatRole
from haystack_experimental.dataclasses.tool import Tool, ToolInvocationError

logger = logging.getLogger(__name__)

_TOOL_INVOCATION_FAILURE = "Tool invocation failed with error: {error}."
_TOOL_NOT_FOUND = "Tool {tool_name} not found in the list of tools. Available tools are: {available_tools}."


class ToolNotFoundException(Exception):
    """
    Exception raised when a tool is not found in the list of available tools.
    """

    pass


@component
class ToolInvoker:
    """
    Invokes tools based on prepared tool calls and returns the results as a list of ChatMessage objects.

    At initialization, the ToolInvoker component is provided with a list of available tools.
    At runtime, the component processes a ChatMessage object containing tool calls and invokes the corresponding tools.
    The results of the tool invocations are returned as a list of ChatMessage objects with tool role.

    Usage example:
    ```python
    from haystack_experimental.dataclasses import ChatMessage, ToolCall, Tool
    from haystack_experimental.components.tools import ToolInvoker

    # Tool definition
    def dummy_weather_function(city: str):
        return f"The weather in {city} is 20 degrees."

    parameters = {"type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]}

    tool = Tool(name="weather_tool",
                description="A tool to get the weather",
                function=dummy_weather_function,
                parameters=parameters)

    # usually, the ChatMessage with tool_calls is generated by a Language Model
    # here we create it manually for demonstration purposes
    tool_call = ToolCall(
        tool_name="weather_tool",
        arguments={"city": "Berlin"}
    )
    message = ChatMessage.from_assistant(tool_calls=[tool_call])


    invoker = ToolInvoker(tools=[tool])
    result = invoker.run(tool_message=message)

    print(result)
    ```

    ```
    >> {'tool_results': [ChatMessage(_role=<ChatRole.TOOL: 'tool'>,
    >>                              _content=[ToolCallResult(result='"The weather in Berlin is 20 degrees."',
    >>                                                       origin=ToolCall(tool_name='weather_tool',
    >>                                                                       arguments={'city': 'Berlin'},
    >>                                                                       id=None))],
    >>                             _meta={})]}
    ```
    """

    def __init__(self, tools: List[Tool], raise_on_failure: bool = True):
        """
        Initialize the ToolInvoker component.

        :param tools:
            A list of tools that can be invoked.
        :param raise_on_failure:
            If True, the component will raise exceptions in case the tool is not found or invocation fails.
            If False, the component will return a ChatMessage from the tool, wrapping the error message.
        """
        if not tools:
            raise ValueError("ToolInvoker requires at least one tool to be provided.")

        self.tools = tools
        self._tools_with_names = {tool.name: tool for tool in tools}
        self.raise_on_failure = raise_on_failure

    @staticmethod
    def _convert_tool_result_to_string(result: Any) -> str:
        """
        Converts the tool invocation result to a string.

        :param result:
            The tool result.
        :returns:
            The string representation of the tool result.
        """
        try:
            return json.dumps(result)
        except TypeError as e:
            logger.warning(
                "Failed to convert tool result to string using `json.dumps`. Error: {error}. Falling back to `str`.",
                error=e,
            )
            return str(result)

    @component.output_types(tool_messages=List[ChatMessage])
    def run(self, message: ChatMessage):
        """
        Processes a ChatMessage containing tool calls and invokes the corresponding tools, if available.

        :param message:
            A ChatMessage from assistant, containing prepared tool calls.
        :returns:
            A dictionary with the key `tool_messages` containing a list of ChatMessage objects with tool role
            that contain the results of the tool invocations.

        :raises ValueError:
            If the message is not from the assistant role or does not contain tool calls.
        :raises ToolNotFoundException:
            If the tool is not found in the list of available tools.
        :raises ToolInvocationError:
            If the tool invocation fails.
        """
        if not message.is_from(ChatRole.ASSISTANT):
            raise ValueError("ToolInvoker only supports messages from the assistant role.")

        tool_calls = message.tool_calls
        if not tool_calls:
            raise ValueError("ToolInvoker only supports messages with tool calls.")

        tool_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            tool_arguments = tool_call.arguments

            if not tool_name in self._tools_with_names:
                msg = _TOOL_NOT_FOUND.format(tool_name=tool_name, available_tools=self._tools_with_names.keys())
                if self.raise_on_failure:
                    raise ToolNotFoundException(msg)
                tool_messages.append(ChatMessage.from_tool(tool_result=msg, origin=tool_call))
                continue

            tool_to_invoke = self._tools_with_names[tool_name]
            try:
                tool_response = tool_to_invoke.invoke(**tool_arguments)
            except ToolInvocationError as e:
                if self.raise_on_failure:
                    raise e
                msg = _TOOL_INVOCATION_FAILURE.format(error=e)
                tool_messages.append(ChatMessage.from_tool(tool_result=msg, origin=tool_call))
                continue

            tool_result_string = self._convert_tool_result_to_string(tool_response)
            tool_messages.append(ChatMessage.from_tool(tool_result=tool_result_string, origin=tool_call))

        return {"tool_messages": tool_messages}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialized_tools = [tool.to_dict() for tool in self.tools]
        return default_to_dict(self, tools=serialized_tools, raise_on_failure=self.raise_on_failure)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolInvoker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        serialized_tools = init_params.get("tools", [])
        deserialized_tools = [Tool.from_dict(tool_data) for tool_data in serialized_tools]
        data["init_parameters"]["tools"] = deserialized_tools
        return default_from_dict(cls, data)
