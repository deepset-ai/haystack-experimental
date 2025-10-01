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
from haystack.tools import create_tool_from_function
from rich.console import Console

from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.components.agents.human_in_the_loop import (
    AlwaysAskPolicy,
    BreakpointConfirmationStrategy,
    HumanInTheLoopStrategy,
    ToolExecutionDecision,
    RichConsoleUI,
)
from haystack_experimental.components.agents.human_in_the_loop.errors import ToolBreakpointException
from haystack_experimental.components.agents.human_in_the_loop.breakpoint import (
    get_tool_calls_and_descriptions,
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
# Using Breakpoint Confirmation Strategy
# ============
# ----
# Step 1: Run agent with breakpoint
# ----
cons.print("\n[bold blue]=== Multiple Sequential Tool Calls Example ===[/bold blue]\n")
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[balance_tool],
    system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
    confirmation_strategies={balance_tool.name: BreakpointConfirmationStrategy()}
)

messages = [
    ChatMessage.from_user(
        "What's the balance of account 56789? If it's lower than $2000, what's the balance of account 12345?"
    )
]
try:
    _ = agent.run(messages=messages)
except ToolBreakpointException as e:
    tool_name = e.break_point.break_point.tool_name
    cons.print("[bold red]Execution paused by Breakpoint Confirmation Strategy for tool:[/bold red]", tool_name)

# ----
# Step 2: Gather user input to optionally update tool parameters before resuming execution
# ----

# ----
# Step 2.1: Extract info to send to "front-end"
# ----
# Sends:
# - tool_calls: List[ToolCall]
# - tool_descriptions: Dict[str, str] {"tool_name": "tool_description"}

# Load the snapshot
possible_snapshots = [Path("pipeline_snapshots") / f for f in os.listdir(Path("pipeline_snapshots"))]
latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
snapshot = load_pipeline_snapshot(latest_snapshot_file)
serialized_tool_calls, tool_descriptions = get_tool_calls_and_descriptions(agent_snapshot=snapshot.agent_snapshot)

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
    tool_execution_decisions.append(
        confirmation_strategy.run(
            tool_name=tc["tool_name"],
            tool_description=tool_descriptions[tc["tool_name"]],
            tool_id=tc["id"],
            tool_params=tc["arguments"],
        )
    )
serialized_teds = [ted.to_dict() for ted in tool_execution_decisions]


# ----
# Step 3: Restart execution with updated snapshot
# ----
new_agent_snapshot = snapshot.agent_snapshot
new_agent_snapshot.tool_execution_decisions = [ToolExecutionDecision.from_dict(ted) for ted in serialized_teds]
# The Breakpoint Confirmation Strategy should trigger again b/c the agent needs to call the tool a second time since
# the bank account balance for account 56789 is below $2000
try:
    _ = agent.run(
        messages=[],
        snapshot=new_agent_snapshot,
    )
except ToolBreakpointException as e:
    tool_name = e.break_point.break_point.tool_name
    cons.print("[bold red]Execution paused by Breakpoint Confirmation Strategy for tool:[/bold red]", tool_name)

# ----
# Step 4: Optionally update tool parameters before resuming execution
# ----

# Load the snapshot
possible_snapshots = [Path("pipeline_snapshots") / f for f in os.listdir(Path("pipeline_snapshots"))]
latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
snapshot = load_pipeline_snapshot(latest_snapshot_file)

# ----
# Step 5: Restart execution after breakpoint
# ----
try:
    result = agent.run(
        messages=[],
        snapshot=snapshot.agent_snapshot,
    )
except ToolBreakpointException:
    raise RuntimeError("Should not hit another breakpoint here!")

last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")
