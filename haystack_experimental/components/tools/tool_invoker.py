# tool_invoker.py

# SPDX-FileCopyrightText: ...
# SPDX-License-Identifier: Apache-2.0
import json
import inspect
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, deserialize_tools_inplace
from haystack.tools.errors import ToolInvocationError

from haystack_experimental.dataclasses.state import State

logger = logging.getLogger(__name__)

_TOOL_INVOCATION_FAILURE = "Tool invocation failed with error: {error}."
_TOOL_NOT_FOUND = "Tool {tool_name} not found in the list of tools. Available tools: {available_tools}."
_TOOL_RESULT_CONVERSION_FAILURE = (
    "Failed to convert tool result to string using '{conversion_function}'. Error: {error}."
)


class ToolNotFoundException(Exception):
    """Raised when a tool is not found in the list of available tools."""
    pass


class StringConversionError(Exception):
    """Raised when the conversion of a tool result to a string fails."""
    pass


@component
class ToolInvoker:
    """
    Invokes tools based on prepared tool calls and returns results as ChatMessage objects.
    Also handles reading/writing from a shared `State`.
    """

    def __init__(
        self,
        tools: List[Tool],
        raise_on_failure: bool = True,
        convert_result_to_json_string: bool = False,
    ):
        """
        :param tools: List of Tool objects that can be invoked.
        :param raise_on_failure: If True, raise exceptions on failure. Otherwise, produce an error ChatMessage.
        :param convert_result_to_json_string: If True, tool results are json.dumps(...) to string. If False, str(...) is used.
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

    def _get_function_parameters(self, func) -> set:
        """Return a set of parameter names from a function's signature."""
        return set(inspect.signature(func).parameters.keys())

    def _prepare_tool_result_message(self, result: Any, tool_call: ToolCall) -> ChatMessage:
        """
        Convert tool result to ChatMessage (with role=TOOL).
        """
        error = False
        if self.convert_result_to_json_string:
            try:
                tool_result_str = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                if self.raise_on_failure:
                    raise StringConversionError(
                        "Failed to convert tool result with `json.dumps`"
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
                        "Failed to convert tool result with `str`"
                    ) from e
                tool_result_str = _TOOL_RESULT_CONVERSION_FAILURE.format(
                    error=e, conversion_function="str"
                )
                error = True

        return ChatMessage.from_tool(
            tool_result=tool_result_str, error=error, origin=tool_call
        )

    def _inject_state_args(
        self, tool: Tool, llm_args: Dict[str, Any], state: State
    ) -> Dict[str, Any]:
        """
        Combine LLM-provided arguments (llm_args) with state-based arguments, respecting:
          - local tool.inputs mappings (if any)
          - function signature name matching
          - LLM overrides state if the same param is present in both
        """
        final_args = dict(llm_args)  # start with LLM-provided
        func_params = self._get_function_parameters(tool.function)

        # If this is a "ComponentTool" (or function-based tool) that has an 'inputs' mapping, use it.
        # Typically, a "Tool" might have .inputs = {"state_key": "tool_param_name"}
        if hasattr(tool, "inputs") and isinstance(tool.inputs, dict):
            # Only pull from state if the LLM did *not* provide a value.
            for state_key, param_name in tool.inputs.items():
                if param_name not in final_args and state.has(state_key):
                    final_args[param_name] = state.get(state_key)
        else:
            # Fallback: auto-match by name if function param is in state, not overridden by LLM
            for param_name in func_params:
                if param_name not in final_args and state.has(param_name):
                    final_args[param_name] = state.get(param_name)

        # ToolCall arguments from the LLM always override state if there's a collision.
        # (We've already copied them in final_args.)
        return final_args

    @staticmethod
    def _merge_tool_outputs(tool: Tool, result: Any, state: State) -> Any:
        """
        Merge a tool result into the global state. If `result` is a dictionary and there's an
        `outputs` mapping on the tool, apply it. Otherwise, do default merges keyed by dictionary keys.

        Return the string that should appear as the final "tool role" message for the LLM.
          - By default, we convert the entire result to a string.
          - If the tool defines "message" in outputs, that "message" becomes the conversation text.
        """
        if not isinstance(result, dict):
            # Not a dict => just treat it as a single output
            return result

        # If the tool has an .outputs mapping with local overrides
        # e.g. tool.outputs = {
        #   "message": {"source": "documents", "handler": docs_to_str},
        #   "documents": {"source": "documents", "handler": None}
        # }
        # Then "message" is special: it forms the chat message text
        # The rest is merged into state
        if hasattr(tool, "outputs") and isinstance(tool.outputs, dict) and tool.outputs:
            message_content = None
            for state_key, config in tool.outputs.items():
                # Where do we pull the data from in `result`?
                # If "source" is given, use that subkey. Otherwise, entire result
                source_key = config.get("source", None)
                output_value = result if source_key is None else result.get(source_key)

                # Apply local handler if any
                handler = config.get("handler", None)

                if state_key == "message":
                    # This is how we produce the final text for the LLM
                    if handler is not None:
                        try:
                            message_content = handler(output_value)
                        except Exception:
                            message_content = f"[Error in message handler for {state_key}]"
                    else:
                        message_content = output_value
                else:
                    # It's a state field => merge
                    # If there's a local custom handler, pass it as override
                    state.set(state_key, output_value, handler_override=handler)

            # If no explicit "message" key, fallback to the entire result
            return message_content if message_content is not None else result
        else:
            # No explicit outputs => each key in result merges into state
            # The entire dict is stringified for the LLM
            for k, v in result.items():
                state.set(k, v)

            return result

    @component.output_types(messages=List[ChatMessage], state=State)
    def run(self, messages: List[ChatMessage], state: Optional[State] = None) -> Dict[str, Any]:
        """
        Look for tool calls in the ChatMessages. For each tool call:
        1. Merge in any needed arguments from the `state`.
        2. Invoke the tool, capturing errors.
        3. Merge returned data back into `state` and produce a ChatMessage with the tool's textual output.

        Returns:
          {
            "messages": <list of ChatMessage with role=TOOL>,
            "state": <updated State>
          }
        """
        if state is None:
            # If none provided, create an empty state
            state = State(schema={})

        tool_messages = []

        for message in messages:
            tool_calls = message.tool_calls
            if not tool_calls:
                continue

            for tool_call in tool_calls:
                tool_name = tool_call.tool_name
                llm_args = tool_call.arguments.copy()

                if tool_name not in self._tools_with_names:
                    msg = _TOOL_NOT_FOUND.format(
                        tool_name=tool_name,
                        available_tools=list(self._tools_with_names.keys()),
                    )
                    if self.raise_on_failure:
                        raise ToolNotFoundException(msg)
                    tool_messages.append(
                        ChatMessage.from_tool(tool_result=msg, origin=tool_call, error=True)
                    )
                    continue

                tool_to_invoke = self._tools_with_names[tool_name]

                # 1) Combine user + state inputs
                final_args = self._inject_state_args(tool_to_invoke, llm_args, state)

                # 2) Invoke the tool
                try:
                    tool_result = tool_to_invoke.invoke(**final_args)
                except ToolInvocationError as e:
                    if self.raise_on_failure:
                        raise
                    msg = _TOOL_INVOCATION_FAILURE.format(error=e)
                    tool_messages.append(
                        ChatMessage.from_tool(tool_result=msg, origin=tool_call, error=True)
                    )
                    continue

                # 3) Merge outputs into state & create a single ChatMessage for the LLM
                try:
                    tool_text = self._merge_tool_outputs(tool_to_invoke, tool_result, state)
                except Exception as e:
                    if self.raise_on_failure:
                        raise
                    tool_text = f"[Merging error: {e}]"

                tool_messages.append(
                    ChatMessage.from_tool(
                        tool_result=tool_text,
                        error=False,
                        origin=tool_call
                    )
                )

        return {"messages": tool_messages, "state": state}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the ToolInvoker.
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
        Deserialize the ToolInvoker.
        """
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        return default_from_dict(cls, data)
