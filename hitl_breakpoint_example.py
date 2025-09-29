# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import os
from pathlib import Path

from haystack.components.agents.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.breakpoints import AgentBreakpoint, ToolBreakpoint
from haystack.tools import create_tool_from_function
from rich.console import Console

from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.components.agents.human_in_the_loop.policies import (
    AlwaysAskPolicy,
)
from haystack_experimental.components.agents.human_in_the_loop.strategies import HumanInTheLoopStrategy
from haystack_experimental.components.agents.human_in_the_loop.user_interfaces import (
    RichConsoleUI,
)


def get_bank_balance(account_id: str) -> str:
    """
    Simulate fetching a bank balance for a given account ID.

    :param account_id: The ID of the bank account.
    :returns:
        A string representing the bank balance.
    """
    return f"Balance for account {account_id} is $1,234.56"


balance_tool = create_tool_from_function(
    function=get_bank_balance,
    name="get_bank_balance",
    description="Get the bank balance for a given account ID.",
)

# Define shared console
cons = Console()

# ============
# Using Only Breakpoint Feature
# ============
# ----
# Step 1: Run agent with breakpoint
# ----
cons.print("\n[bold blue]=== Using Only Breakpoint Feature ===[/bold blue]\n")
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[balance_tool],
    system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
)

# This breakpoint will cause a BreakpointException to raise after the tool is selected but before execution
agent_break_point = AgentBreakpoint(
    agent_name="agent",
    break_point=ToolBreakpoint(
        tool_name=balance_tool.name,
        # NOTE: Would be nice to set component_name to "tool_invoker" by default
        component_name="tool_invoker",
        visit_count=0,
        snapshot_file_path="pipeline_snapshots",
    ),
)
messages = [ChatMessage.from_user("What's the balance of account 56789?")]
try:
    _ = agent.run(messages=messages, break_point=agent_break_point)
except BreakpointException:
    cons.print("[bold red]Execution paused at breakpoint.[/bold red]")

# NOTE: Opened issue to make it easier to find the latest snapshot: https://github.com/deepset-ai/haystack/issues/9828
possible_snapshots = [Path("pipeline_snapshots") / f for f in os.listdir(Path("pipeline_snapshots"))]
latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
snapshot = load_pipeline_snapshot(latest_snapshot_file)


# ----
# Step 2: Gather user input to optionally update tool parameters before resuming execution
# ----

# ----
# Step 2.1: Extract info to send to "front-end"
# ----
# Sends:
# - tool_calls: List[ToolCall]
# - tool_descriptions: Dict[str, str] {"tool_name": "tool_description"}

# Create the list of tool calls to send
serialized_tool_call_messages = snapshot.agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["messages"]
tool_call_messages = [ChatMessage.from_dict(m) for m in serialized_tool_call_messages]
tool_calls = []
for msg in tool_call_messages:
    if msg.tool_calls:
        tool_calls.extend(msg.tool_calls)
serialized_tool_calls = [tc.to_dict() for tc in tool_calls]
# TODO The arguments in serialized_tool_calls are not fully correct. Missing the injection from inputs_from_state.

# Create the list of tools to send
serialized_tools = snapshot.agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["tools"]

# Create the dict of tool descriptions to send
tool_descriptions = {t["data"]["name"]: t["data"]["description"] for t in serialized_tools}

# ----
# Step 2.1: Mimic the front-end
# ----
# Receives:
# - tool_calls: List[ToolCall]
# - tool_descriptions: Dict[str, str] {"tool_name": "tool_description"}
# - snapshot_id: str
# Sends back:
# - tool_execution_decisions: List[ToolExecutionDecision]
# - snapshot_id: str --> Needed to know which snapshot to load
confirmation_strategy = HumanInTheLoopStrategy(
    confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI(console=cons)
)
tool_execution_decisions = []
for tc in serialized_tool_calls:
    ted = confirmation_strategy.run(
        tool_name=tc["tool_name"],
        tool_description=tool_descriptions[tc["tool_name"]],
        # TODO Could easily add tool_id here since we have the ToolCall
        # tool_id=tc["id"],
        tool_params=tc["arguments"],
    )
    tool_execution_decisions.append(ted)

# ----
# Step 2.2: Update tool call messages and state based on feedback from the "front-end"
# ----
# Receives:
# - tool_execution_decisions: List[ToolExecutionDecision]
# - snapshot_id: str --> Needed to know which snapshot to load
# TODO We should be updating the existing tool call ChatMessage instead of creating new ones otherwise we lose info
#      Semi-complicated since the ChatMessage can have multiple tool calls inside and the tool_execution_decisions
#      is a flat list with only tool_name as a linking element. --> linking element should be tool_id b/c a tool can
#      be called multiple times in a single tool call message.
tool_id_to_tool_call = {tc["id"]: tc for tc in serialized_tool_calls}
tool_name_to_tool_call = {tc["tool_name"]: tc for tc in serialized_tool_calls}

new_tool_call_messages = []
additional_state_messages = []
for ted in tool_execution_decisions:
    if ted.tool_id:
        tool_call = tool_id_to_tool_call[ted.tool_id]
    else:
        tool_call = tool_name_to_tool_call[ted.tool_name]
    if ted.execute:
        # Covers confirm and modify cases
        if tool_call["arguments"] != ted.final_tool_params:
            # In the modify case we add a user message explaining the modification otherwise the LLM won't know why the
            # tool parameters changed and will likely just try and call the tool again with the original parameters.
            new_tool_call_messages.append(
                ChatMessage.from_user(
                    text=(
                        f"The parameters for tool '{tool_call['tool_name']}' were updated by the user to:\n"
                        f"{ted.final_tool_params}"
                    )
                )
            )
        new_tool_call_messages.append(
            ChatMessage.from_assistant(
                tool_calls=[replace(ToolCall.from_dict(tool_call), arguments=ted.final_tool_params)]
            )
        )
    else:
        # Reject case
        # We create a tool call result message using the feedback from the confirmation strategy
        # Then we move both the tool call message and the tool call result message to the chat history in State
        additional_state_messages.append(ChatMessage.from_assistant(tool_calls=[ToolCall.from_dict(tool_call)]))
        additional_state_messages.append(
            ChatMessage.from_tool(
                tool_result=ted.feedback or "",
                origin=ToolCall.from_dict(tool_call),
                error=True,
            )
        )

# Modify the chat history in state to handle the rejection cases
# 1. Move the tool call message and tool call result message pairs to right after last user message
# 2. Leave all remaining tool call messages (i.e. the ones that were confirmed or modified) at the end of the chat
serialized_state = snapshot.agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["state"]
state = State.from_dict(serialized_state)
chat_history = state.get("messages")
last_user_msg_idx = max(i for i, m in enumerate(chat_history) if m.is_from("user"))
new_chat_history = chat_history[: last_user_msg_idx + 1] + additional_state_messages + new_tool_call_messages
state.set(key="messages", value=new_chat_history, handler_override=replace_values)

# Update the snapshot with the new tool call messages and updated state
snapshot.agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["messages"] = [
    msg.to_dict() for msg in new_tool_call_messages
]
snapshot.agent_snapshot.component_inputs["tool_invoker"]["serialized_data"]["state"] = state.to_dict()

# ----
# Step 3: Restart execution after breakpoint with updated snapshot
# ----
# NOTE: We extended the Agent in haystack-experimental to allow running with a snapshot and a breakpoint
result = agent.run(
    # NOTE: Messages are still required but are ignored when passing in a snapshot
    messages=[],
    snapshot=snapshot.agent_snapshot,
    # This break point shouldn't trigger again, since the agent will exit after the next llm call to give the
    # final answer.
    # TODO Add a convenience function to return a new breakpoint with an incremented visit count ??
    break_point=replace(
        snapshot.agent_snapshot.break_point,
        break_point=replace(
            snapshot.agent_snapshot.break_point.break_point,
            visit_count=snapshot.agent_snapshot.break_point.break_point.visit_count + 1,
        ),
    ),
)
last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")
