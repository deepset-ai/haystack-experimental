# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools.errors import ToolInvocationError

from haystack_experimental.dataclasses.state import State
from haystack_experimental.tools import Tool, deserialize_tools_inplace
from haystack_experimental.tools.component_tool import ComponentTool

logger = logging.getLogger(__name__)


class ToolInvokerError(Exception):
    """Base exception class for ToolInvoker errors."""
    def __init__(self, message: str):
        super().__init__(message)


class ToolNotFoundException(ToolInvokerError):
    """Exception raised when a tool is not found in the list of available tools."""
    def __init__(self, tool_name: str, available_tools: List[str]):
        message = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
        super().__init__(message)


class StringConversionError(ToolInvokerError):
    """Exception raised when the conversion of a tool result to a string fails."""
    def __init__(self, tool_name: str, conversion_function: str, error: Exception):
        message = f"Failed to convert tool result from tool {tool_name} using '{conversion_function}'. Error: {error}"
        super().__init__(message)


class ToolOutputMergeError(ToolInvokerError):
    """Exception raised when merging tool outputs into state fails."""
    pass


@component
class ToolInvoker:
    """
    Invokes tools based on prepared tool calls and returns the results as a list of ChatMessage objects.

    Also handles reading/writing from a shared `State`.
    At initialization, the ToolInvoker component is provided with a list of available tools.
    At runtime, the component processes a list of ChatMessage object containing tool calls
    and invokes the corresponding tools.
    The results of the tool invocations are returned as a list of ChatMessage objects with tool role.

    Usage example:
    ```python
    from haystack.dataclasses import ChatMessage, ToolCall, Tool
    from haystack.components.tools import ToolInvoker

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

    # Usually, the ChatMessage with tool_calls is generated by a Language Model
    # Here, we create it manually for demonstration purposes
    tool_call = ToolCall(
        tool_name="weather_tool",
        arguments={"city": "Berlin"}
    )
    message = ChatMessage.from_assistant(tool_calls=[tool_call])

    # ToolInvoker initialization and run
    invoker = ToolInvoker(tools=[tool])
    result = invoker.run(messages=[message])

    print(result)
    ```

    ```
    >>  {
    >>      'tool_messages': [
    >>          ChatMessage(
    >>              _role=<ChatRole.TOOL: 'tool'>,
    >>              _content=[
    >>                  ToolCallResult(
    >>                      result='"The weather in Berlin is 20 degrees."',
    >>                      origin=ToolCall(
    >>                          tool_name='weather_tool',
    >>                          arguments={'city': 'Berlin'},
    >>                          id=None
    >>                      )
    >>                  )
    >>              ],
    >>              _meta={}
    >>          )
    >>      ]
    >>  }
    ```
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
        :raises ValueError:
            If no tools are provided or if duplicate tool names are found.
        """
        if not tools:
            raise ValueError("ToolInvoker requires at least one tool.")
        tool_names = [tool.name for tool in tools]
        duplicates = {name for name in tool_names if tool_names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate tool names found: {duplicates}")

        self.tools = tools
        self._tools_with_names = dict(zip(tool_names, tools))
        self.raise_on_failure = raise_on_failure
        self.convert_result_to_json_string = convert_result_to_json_string

    def _handle_error(self, error: Exception) -> str:
        """
        Handles errors by logging and either raising or returning a fallback error message.

        :param error: The exception instance.
        :returns: The fallback error message when `raise_on_failure` is False.
        :raises: The provided error if `raise_on_failure` is True.
        """
        logger.error("{error_exception}", error_exception=error)
        if self.raise_on_failure:
            # We re-raise the original error maintaining the exception chain
            raise error
        return str(error)

    def _prepare_tool_result_message(self, result: Any, tool_call: ToolCall) -> ChatMessage:
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
        try:
            if self.convert_result_to_json_string:
                # We disable ensure_ascii so special chars like emojis are not converted
                tool_result_str = json.dumps(result, ensure_ascii=False)
            else:
                tool_result_str = str(result)
        except Exception as e:
            conversion_method = "json.dumps" if self.convert_result_to_json_string else "str"
            try:
                tool_result_str = self._handle_error(StringConversionError(tool_call.tool_name, conversion_method, e))
                error = True
            except StringConversionError as conversion_error:
                # If _handle_error re-raises, this properly preserves the chain
                raise conversion_error from e

        return ChatMessage.from_tool(tool_result=tool_result_str, error=error, origin=tool_call)

    @staticmethod
    def _inject_state_args(tool: Tool, llm_args: Dict[str, Any], state: State) -> Dict[str, Any]:
        """
        Combine LLM-provided arguments (llm_args) with state-based arguments.

        Tool arguments take precedence in the following order:
          - LLM overrides state if the same param is present in both
          - local tool.inputs mappings (if any)
          - function signature name matching
        """
        final_args = dict(llm_args)  # start with LLM-provided

        # ComponentTool wraps the function with a function that accepts kwargs, so we need to look at input sockets
        # to find out which parameters the tool accepts.
        if isinstance(tool, ComponentTool):
            func_params = set(tool._component.__haystack_input__._sockets_dict.keys())
        else:
            func_params = set(inspect.signature(tool.function).parameters.keys())

        # Determine the source of parameter mappings (explicit tool inputs or direct function parameters)
        # Typically, a "Tool" might have .inputs_from_state = {"state_key": "tool_param_name"}
        if hasattr(tool, "inputs_from_state") and isinstance(tool.inputs_from_state, dict):
            param_mappings = tool.inputs_from_state
        else:
            param_mappings = {name: name for name in func_params}

        # Populate final_args from state if not provided by LLM
        for state_key, param_name in param_mappings.items():
            if param_name not in final_args and state.has(state_key):
                final_args[param_name] = state.get(state_key)

        return final_args

    def _merge_tool_outputs(self, tool: Tool, result: Any, state: State) -> Any:
        """
        Merges the tool result into the global state and determines the response message.

        This method processes the output of a tool execution and integrates it into the global state.
        It also determines what message, if any, should be returned for further processing in a conversation.

        Processing Steps:
        1. If `result` is not a dictionary, nothing is stored into state and the full `result` is returned.
        2. If the `tool` does not define an `outputs` mapping nothing is stored into state.
           The return value in this case is simply the full `result` dictionary.
        3. If the tool defines an `outputs` mapping (a dictionary describing how the tool's output should be processed),
           the method delegates to `_handle_tool_outputs` to process the output accordingly.
           This allows certain fields in `result` to be mapped explicitly to state fields or formatted using custom
           handlers.

        :param tool: Tool instance containing optional `outputs` mapping to guide result processing.
        :param result: The output from tool execution. Can be a dictionary, or any other type.
        :param state: The global State object to which results should be merged.
        :returns: Three possible values:
            - A string message for conversation
            - The merged result dictionary
            - Or the raw result if not a dictionary
        """
        # If result is not a dictionary, return it as the output message.
        if not isinstance(result, dict):
            return result

        # If there is no specific `outputs` mapping, we just return the full result
        if not hasattr(tool, "outputs_to_state") or not isinstance(tool.outputs_to_state, dict):
            return result

        # Handle tool outputs with specific mapping for message and state updates
        return self._handle_tool_outputs(tool.outputs_to_state, result, state)

    @staticmethod
    def _handle_tool_outputs(outputs: dict, result: dict, state: State) -> Union[dict, str]:
        """
        Handles the `outputs` mapping from the tool and updates the state accordingly.

        :param outputs: Mapping of outputs from the tool.
        :param result: Result of the tool execution.
        :param state: Global state to merge results into.
        :returns: Final message for LLM or the entire result.
        """
        message_content = None

        for state_key, config in outputs.items():
            # Get the source key from the output config, otherwise use the entire result
            source_key = config.get("source", None)
            output_value = result if source_key is None else result.get(source_key)

            # Get the handler function, if any
            handler = config.get("handler", None)

            if state_key == "message":
                # Handle the message output separately
                if handler is not None:
                    message_content = handler(output_value)
                else:
                    message_content = str(output_value)
            else:
                # Merge other outputs into the state
                state.set(state_key, output_value, handler_override=handler)

        # If no "message" key was found, return the result or message content
        return message_content if message_content is not None else result

    @component.output_types(tool_messages=List[ChatMessage], state=State)
    def run(self, messages: List[ChatMessage], state: Optional[State] = None) -> Dict[str, Any]:
        """
        Processes ChatMessage objects containing tool calls and invokes the corresponding tools, if available.

        :param messages:
            A list of ChatMessage objects.
        :param state: The runtime state that should be used by the tools.
        :returns:
            A dictionary with the key `tool_messages` containing a list of ChatMessage objects with tool role.
            Each ChatMessage objects wraps the result of a tool invocation.

        :raises ToolNotFoundException:
            If the tool is not found in the list of available tools and `raise_on_failure` is True.
        :raises ToolInvocationError:
            If the tool invocation fails and `raise_on_failure` is True.
        :raises StringConversionError:
            If the conversion of the tool result to a string fails and `raise_on_failure` is True.
        :raises ToolOutputMergeError:
            If merging tool outputs into state fails and `raise_on_failure` is True.
        """
        if state is None:
            state = State(schema={})

        # Only keep messages with tool calls
        messages_with_tool_calls = [message for message in messages if message.tool_calls]

        tool_messages = []
        for message in messages_with_tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.tool_name

                # Check if the tool is available, otherwise return an error message
                if tool_name not in self._tools_with_names:
                    error_message = self._handle_error(
                        ToolNotFoundException(tool_name, list(self._tools_with_names.keys()))
                    )
                    tool_messages.append(
                        ChatMessage.from_tool(tool_result=error_message, origin=tool_call, error=True)
                    )
                    continue

                tool_to_invoke = self._tools_with_names[tool_name]

                # 1) Combine user + state inputs
                llm_args = tool_call.arguments.copy()
                final_args = self._inject_state_args(tool_to_invoke, llm_args, state)

                # 2) Invoke the tool
                try:
                    tool_result = tool_to_invoke.invoke(**final_args)
                except ToolInvocationError as e:
                    error_message = self._handle_error(e)
                    tool_messages.append(ChatMessage.from_tool(tool_result=error_message, origin=tool_call, error=True))
                    continue

                # 3) Merge outputs into state & create a single ChatMessage for the LLM
                try:
                    tool_text = self._merge_tool_outputs(tool_to_invoke, tool_result, state)
                except Exception as e:
                    try:
                        error_message = self._handle_error(
                            ToolOutputMergeError(f"Failed to merge tool outputs from tool {tool_name} into State: {e}")
                        )
                        tool_messages.append(
                            ChatMessage.from_tool(tool_result=error_message, origin=tool_call, error=True)
                        )
                        continue
                    except ToolOutputMergeError as propagated_e:
                        # Re-raise with proper error chain
                        raise propagated_e from e

                tool_messages.append(self._prepare_tool_result_message(result=tool_text, tool_call=tool_call))

        return {"tool_messages": tool_messages, "state": state}

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
