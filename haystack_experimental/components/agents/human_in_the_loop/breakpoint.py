# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses.breakpoints import AgentSnapshot, ToolBreakpoint
from haystack.utils import _deserialize_value_with_schema

from haystack_experimental.components.agents.human_in_the_loop.strategies import _prepare_tool_args


def get_tool_calls_and_descriptions_from_snapshot(
    agent_snapshot: AgentSnapshot, breakpoint_tool_only: bool = True
) -> tuple[list[dict], dict[str, str]]:
    """
    Extract tool calls and tool descriptions from an AgentSnapshot.

    By default, only the tool call that caused the breakpoint is processed and its arguments are reconstructed.
    This is useful for scenarios where you want to present the relevant tool call and its description
    to a human for confirmation before execution.

    :param agent_snapshot: The AgentSnapshot from which to extract tool calls and descriptions.
    :param breakpoint_tool_only: If True, only the tool call that caused the breakpoint is returned. If False, all tool
        calls are returned.
    :returns:
        A tuple containing a list of tool call dictionaries and a dictionary of tool descriptions
    """
    break_point = agent_snapshot.break_point.break_point
    if not isinstance(break_point, ToolBreakpoint):
        raise ValueError("The provided AgentSnapshot does not contain a ToolBreakpoint.")

    tool_caused_break_point = break_point.tool_name

    # Deserialize the tool invoker inputs from the snapshot
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
    tool_descriptions = {}
    updated_tool_calls = []
    for tc in serialized_tcs:
        # Only process the tool that caused the breakpoint if breakpoint_tool_only is True
        if breakpoint_tool_only and tc["tool_name"] != tool_caused_break_point:
            continue

        final_args = _prepare_tool_args(
            tool=tool_name_to_tool[tc["tool_name"]],
            tool_call_arguments=tc["arguments"],
            state=state,
            streaming_callback=tool_invoker_inputs.get("streaming_callback", None),
            enable_streaming_passthrough=tool_invoker_inputs.get("enable_streaming_passthrough", False),
        )
        updated_tool_calls.append({**tc, "arguments": final_args})
        tool_descriptions[tc["tool_name"]] = tool_name_to_tool[tc["tool_name"]].description

    return updated_tool_calls, tool_descriptions
