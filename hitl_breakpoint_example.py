# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import os
from pathlib import Path

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentBreakpoint, ToolBreakpoint
from haystack.tools import Tool, create_tool_from_function
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
# - tools: Dict[str, Tool] --> Needs to change to tool_descriptions
# - snapshot_id: str
# Sends back:
# - tool_execution_decisions: List[ToolExecutionDecision]
# - snapshot_id: str --> Needed to know which snapshot to load
# TODO Not ideal to have to reconstruct the tool, better to update confirmation strategy to need only tool.name and
#      tool.description explicitly
tools = {t["data"]["name"]: Tool.from_dict(**t) for t in serialized_tools}
confirmation_strategy = HumanInTheLoopStrategy(
    confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI(console=cons)
)
tool_execution_decisions = []
for tc in serialized_tool_calls:
    ted = confirmation_strategy.run(
        tool=tools[tc["tool_name"]],
        tool_params=tc["arguments"],
    )
    tool_execution_decisions.append(ted)

# ----
# Step 2.2: Update tool call messages and state based on feedback from the "front-end"
# ----
# Receives:
# - tool_execution_decisions: List[ToolExecutionDecision]
# - snapshot_id: str --> Needed to know which snapshot to load
tool_name_to_tc_message = {tc.tool_name: tc for tc in tool_calls}
new_tool_call_messages = []
additional_state_messages = []
for ted in tool_execution_decisions:
    if ted.execute:
        # Covers confirm and modify cases
        # NOTE: The modification case slightly differs from not using break points since here we modify the
        #       arguments directly on the tool call message. In the non-breakpoint case we leave the tool call message
        #       as-is and add a string to the tool call result message explaining the modification.
        new_tool_call_messages.append(replace(tool_name_to_tc_message[ted.tool_name], arguments=ted.final_tool_params))
    else:
        # Reject case
        # We create a tool call result message using the feedback from the confirmation strategy
        # Then we move both the tool call message and the tool call result message to the chat history in State
        additional_state_messages.append(tool_name_to_tc_message[ted.tool_name])
        additional_state_messages.append(
            ChatMessage.from_tool(
                tool_result=ted.feedback or "",
                origin=tool_name_to_tc_message[ted.tool_name],
                error=True,
            )
        )

serialized_state = snapshot.agent_snapshot.component_inputs["tool_invoker"]["state"]


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
