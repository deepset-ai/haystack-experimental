# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List

from haystack import component, default_from_dict, default_to_dict

from haystack_experimental.dataclasses import ChatMessage, ChatRole
from haystack_experimental.dataclasses.tool import Tool, ToolInvocationError

_TOOL_NAME_FAILURE = "Tool {tool_name} not found in the list of tools."

_TOOL_RUN_FAILURE = "Following error occurred while attempting to run the tool: {error}"


@component
class ToolInvoker:
    """
    The `ToolInvoker` class processes a list of chat messages and invokes the appropriate tools when needed.

    `ToolInvoker` expects a list of `ChatMessage` objects and utilizes the `tool_call` present in these messages
    to trigger the corresponding tools from a provided list of `Tools`.

    """

    def __init__(self, available_tools: List[Tool], raise_on_failure: bool = True):
        """
        Initialize the ToolInvoker component.

        :param available_tools: A list of available tools.
        :param raise_on_failure: If True, the component will raise an exception if a tool fails to run.
        """
        self._available_tools = available_tools
        self._tool_names = {tool.name for tool in available_tools}
        self._raise_on_failure = raise_on_failure

    @component.output_types(tool_results=List[ChatMessage])
    def run(self, tool_message: ChatMessage):
        """
        Processes `tool_message` that contains tool calls and invoke the corresponding tools.

        :param tool_message: A `ChatMessage` object containing tool_calls to be invoked.
        :returns: Returns a list of ChatMessages.

        :raises ValueError: If the `ChatMessage` does not originate from `ASSISTANT` or contains no `tool_calls`.
        :raises Exception: If a tool specified in the `tool_call` is not found in the `available_tools`.
        """
        tool_results = []

        if not tool_message.is_from(ChatRole.ASSISTANT):
            raise ValueError("ToolInvoker only supports messages from the assistant role.")

        tool_calls = tool_message.tool_calls
        if not tool_calls:
            raise ValueError("ToolInvoker only supports messages with tool calls.")

        for tool_call in tool_calls:
            tool_name = tool_call.tool_name
            tool_arguments = tool_call.arguments
            if tool_name in self._tool_names:
                tool_to_call = next(tool for tool in self._available_tools if tool.name == tool_name)
                try:
                    tool_response = tool_to_call.invoke(**tool_arguments)
                    tool_results.append(ChatMessage.from_tool(tool_result=json.dumps(tool_response), origin=tool_call))

                except ToolInvocationError as exception:
                    if self._raise_on_failure:
                        raise exception
                    tool_results.append(
                        ChatMessage.from_tool(tool_result=_TOOL_RUN_FAILURE.format(error=exception), origin=tool_call)
                    )
            else:
                raise Exception(_TOOL_NAME_FAILURE.format(tool_name=tool_name))
        return {"tool_results": tool_results}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialized_tools = [tool.to_dict() for tool in self._available_tools]
        return default_to_dict(self, available_tools=serialized_tools, raise_on_failure=self._raise_on_failure)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolInvoker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        serialized_tools = data.get("init_parameters", {}).get("available_tools")
        available_tools = [Tool.from_dict(tool_data) for tool_data in serialized_tools]
        data["init_parameters"]["available_tools"] = available_tools
        return default_from_dict(cls, data)
