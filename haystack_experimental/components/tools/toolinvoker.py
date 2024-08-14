# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import List

from haystack import component

from haystack_experimental.dataclasses import ChatMessage, ChatRole
from haystack_experimental.dataclasses.tool import Tool, ToolInvocationError

_FUNCTION_NAME_FAILURE = (
    "I'm sorry, I tried to run a function that did not exist. Would you like me to correct it and try again?"
)
_FUNCTION_RUN_FAILURE = "Seems there was an error while running the function: {error}"


@component
class ToolInvoker:
    def __init__(self, tools: List[Tool], raise_on_failure: bool = True):
        self.tools = tools
        self._tool_names = {tool.name for tool in tools}
        self.raise_on_failure = raise_on_failure

    @component.output_types(tool_results=List[ChatMessage])
    def run(self, tool_message: ChatMessage):
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
                tool_to_call = next(tool for tool in self.tools if tool.name == tool_name)
                try:
                    tool_response = tool_to_call.invoke(**tool_arguments)
                    tool_results.append(ChatMessage.from_tool(tool_result=json.dumps(tool_response), origin=tool_call))

                except ToolInvocationError as e:
                    if self.raise_on_failure:
                        raise e
                    tool_results.append(
                        ChatMessage.from_tool(tool_result=_FUNCTION_RUN_FAILURE.format(error=e), origin=tool_call)
                    )
            else:
                raise Exception(f"Tool {tool_name} not found in the list of tools.")
        return {"tool_results": tool_results}
