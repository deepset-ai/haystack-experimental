# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses.breakpoints import AgentSnapshot
from haystack.utils import _deserialize_value_with_schema

from haystack_experimental.components.agents.human_in_the_loop.strategies import _prepare_tool_args


# TODO This potentially could live in hayhooks instead. At best it's a utility function that is only used in the
#      BreakpointConfirmationStrategy example and could be used in hayhooks to enable human-in-the-loop for agents.
def _get_tool_calls_and_descriptions(agent_snapshot: AgentSnapshot) -> tuple[list[dict], dict[str, str]]:
    """
    Extract tool calls and tool descriptions from an AgentSnapshot.

    This is useful for scenarios where you want to present the tool calls and their descriptions
    to a human for confirmation before execution.

    :param agent_snapshot: The AgentSnapshot from which to extract tool calls and descriptions.
    :return: A tuple containing a list of tool call dictionaries and a dictionary of tool descriptions
    """
    tool_invoker_inputs = _deserialize_value_with_schema(agent_snapshot.component_inputs["tool_invoker"])
    tool_call_messages = tool_invoker_inputs["messages"]
    state = tool_invoker_inputs["state"]
    tool_name_to_tool = {t.name: t for t in tool_invoker_inputs["tools"]}

    tool_calls = []
    for msg in tool_call_messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    serialized_tcs = [tc.to_dict() for tc in tool_calls]

    # Reconstruct the final arguments for each tool call
    updated_tool_calls = []
    for tc in serialized_tcs:
        final_args = _prepare_tool_args(
            tool=tool_name_to_tool[tc["tool_name"]],
            tool_call_arguments=tc["arguments"],
            state=state,
            streaming_callback=tool_invoker_inputs.get("streaming_callback", None),
            enable_streaming_passthrough=tool_invoker_inputs.get("enable_streaming_passthrough", False)
        )
        updated_tool_calls.append({**tc, "arguments": final_args})

    # Create the dict of tool descriptions to send
    tool_descriptions = {t.name: t.description for t in tool_invoker_inputs["tools"]}
    return serialized_tcs, tool_descriptions
