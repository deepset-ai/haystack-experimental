# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
# type: ignore
import json
import inspect
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging

from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, deserialize_tools_inplace
from haystack.tools.errors import ToolInvocationError

from haystack_experimental.components.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)

_TOOL_INVOCATION_FAILURE = "Tool invocation failed with error: {error}."
_TOOL_NOT_FOUND = "Tool {tool_name} not found in the list of tools. Available tools are: {available_tools}."
_TOOL_RESULT_CONVERSION_FAILURE = "Failed to convert tool result to string using '{conversion_function}'. Error: {error}."


class ToolNotFoundException(Exception):
    """
    Exception raised when a tool is not found in the list of available tools.
    """

    pass


class StringConversionError(Exception):
    """
    Exception raised when the conversion of a tool result to a string fails.
    """

    pass


@component
class ToolInvoker:
    """
    Invokes tools based on prepared tool calls and returns the results as a list of ChatMessage objects.
    Can optionally handle context and output variables that are passed to the tools if their function
    signatures accept them.
    """

    def __init__(
        self,
        tools: List[Tool],
        raise_on_failure: bool = True,
        convert_result_to_json_string: bool = False,
    ):
        """
        Initialize the ToolInvoker component.

        :param tools:
            A list of tools that can be invoked.
        :param raise_on_failure:
            If True, the component will raise an exception in case of errors
            (tool not found, tool invocation errors, tool result conversion errors).
            If False, the component will return a ChatMessage object with `error=True`
            and a description of the error in `result`.
        :param convert_result_to_json_string:
            If True, the tool invocation result will be converted to a string using `json.dumps`.
            If False, the tool invocation result will be converted to a string using `str`.
        """
        if not tools:
            raise ValueError("ToolInvoker requires at least one tool to be provided.")
        tool_names = [tool.name for tool in tools]
        duplicate_tool_names = {
            name for name in tool_names if tool_names.count(name) > 1
        }
        if duplicate_tool_names:
            raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")

        self.tools = tools
        self._tools_with_names = dict(zip(tool_names, tools))
        self.raise_on_failure = raise_on_failure
        self.convert_result_to_json_string = convert_result_to_json_string

    def _prepare_tool_result_message(
        self, result: Any, tool_call: ToolCall
    ) -> ChatMessage:
        """
        Prepares a ChatMessage with the result of a tool invocation.

        :param result:
            The tool result.
        :returns:
            A ChatMessage object containing the tool result as a string.

        :raises
            StringConversionError: If the conversion of the tool result to a string fails
            and `raise_on_failure` is True.
        """
        error = False

        if self.convert_result_to_json_string:
            try:
                tool_result_str = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                if self.raise_on_failure:
                    raise StringConversionError(
                        "Failed to convert tool result to string using `json.dumps`"
                    ) from e
                tool_result_str = _TOOL_RESULT_CONVERSION_FAILURE.format(
                    error=e, conversion_function="json.dumps"
                )
                error = True
        else:
            try:
                tool_result_str = str(result)
            except Exception as e:
                if self.raise_on_failure:
                    raise StringConversionError(
                        "Failed to convert tool result to string using `str`"
                    ) from e
                tool_result_str = _TOOL_RESULT_CONVERSION_FAILURE.format(
                    error=e, conversion_function="str"
                )
                error = True

        return ChatMessage.from_tool(
            tool_result=tool_result_str, error=error, origin=tool_call
        )

    def _get_function_parameters(self, func) -> set:
        """
        Get the parameter names from a function's signature.

        :param func: The function to inspect
        :returns: A set of parameter names
        """
        return set(inspect.signature(func).parameters.keys())

    @component.output_types(messages=List[ChatMessage], context=ToolContext)
    def run(
        self,
        messages: List[ChatMessage],
        context: Optional[ToolContext] = None,
    ) -> Dict[str, Any]:
        """
        Processes ChatMessage objects containing tool calls and invokes the corresponding tools, if available.

        :param messages:
            A list of ChatMessage objects.
        :param context: Additional input and output parameters that will be passed to all tools that accept
            a 'ctx' parameter.
        :returns:
            A dictionary with the keys:
            - `messages`: list of ChatMessage objects with tool role
            - `output_variables`: dictionary containing any modifications made by the tools

        :raises ToolNotFoundException:
            If the tool is not found in the list of available tools and `raise_on_failure` is True.
        :raises ToolInvocationError:
            If the tool invocation fails and `raise_on_failure` is True.
        :raises StringConversionError:
            If the conversion of the tool result to a string fails and `raise_on_failure` is True.
        """
        tool_messages = []

        for message in messages:
            tool_calls = message.tool_calls
            if not tool_calls:
                continue

            for tool_call in tool_calls:
                tool_name = tool_call.tool_name
                tool_arguments = tool_call.arguments.copy()

                if tool_name not in self._tools_with_names:
                    msg = _TOOL_NOT_FOUND.format(
                        tool_name=tool_name,
                        available_tools=self._tools_with_names.keys(),
                    )
                    if self.raise_on_failure:
                        raise ToolNotFoundException(msg)
                    tool_messages.append(
                        ChatMessage.from_tool(
                            tool_result=msg, origin=tool_call, error=True
                        )
                    )
                    continue

                tool_to_invoke = self._tools_with_names[tool_name]
                func_params = self._get_function_parameters(tool_to_invoke.function)

                # Add context if the function accepts it
                if "ctx" in func_params:
                    tool_arguments["ctx"] = context

                try:
                    tool_result = tool_to_invoke.invoke(**tool_arguments)
                except ToolInvocationError as e:
                    if self.raise_on_failure:
                        raise e
                    msg = _TOOL_INVOCATION_FAILURE.format(error=e)
                    tool_messages.append(
                        ChatMessage.from_tool(
                            tool_result=msg, origin=tool_call, error=True
                        )
                    )
                    continue

                tool_message = self._prepare_tool_result_message(tool_result, tool_call)
                tool_messages.append(tool_message)

        return {"messages": tool_messages, "context": context}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialized_tools = [tool.to_dict() for tool in self.tools]
        return default_to_dict(
            self,
            tools=serialized_tools,
            raise_on_failure=self.raise_on_failure,
            convert_result_to_json_string=self.convert_result_to_json_string,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolInvoker":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        return default_from_dict(cls, data)
