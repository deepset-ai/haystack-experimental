# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentSnapshot

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ToolExecutionDecision


# TODO This potentially could live in hayhooks instead. At best it's a utility function that is only used in the
#      BreakpointConfirmationStrategy example and could be used in hayhooks to enable human-in-the-loop for agents.
def _get_tool_calls_and_descriptions(agent_snapshot: AgentSnapshot) -> tuple[list[dict], dict[str, str]]:
    # Create the list of tool calls to send
    serialized_tool_call_messages = agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["messages"]
    tool_call_messages = [ChatMessage.from_dict(m) for m in serialized_tool_call_messages]
    tool_calls = []
    for msg in tool_call_messages:
        if msg.tool_calls:
            tool_calls.extend(msg.tool_calls)
    serialized_tcs = [tc.to_dict() for tc in tool_calls]
    # TODO The arguments in serialized_tool_calls are not fully correct. Missing the injection from inputs_from_state.

    # Create the dict of tool descriptions to send
    serialized_tools = agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["tools"]
    tool_descriptions = {t["data"]["name"]: t["data"]["description"] for t in serialized_tools}
    return serialized_tcs, tool_descriptions
