# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
# Using Confirmation Strategy with Breakpoint
# ============
# ----
# Step 1: Run agent with breakpoint
# ----
cons.print("\n[bold blue]=== Using Only Breakpoint Feature ===[/bold blue]\n")
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[balance_tool],
    system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
    # This breakpoint will cause a ToolBreakpointException to raised after the tool is selected but before execution
    confirmation_strategies={balance_tool.name: BreakpointConfirmationStrategy()}
)

messages = [ChatMessage.from_user("What's the balance of account 56789?")]
try:
    _ = agent.run(messages=messages)
except BreakpointException:
    cons.print("[bold red]Execution paused at breakpoint.[/bold red]")

# ----
# Step 2: Gather user input to optionally update tool parameters before resuming execution
# ----

# ----
# Step 2.1: Extract info to send to "front-end"
# ----
# Sends:
# - tool_calls: List[ToolCall]
# - tool_descriptions: Dict[str, str] {"tool_name": "tool_description"}

# NOTE: Opened issue to make it easier to find the latest snapshot: https://github.com/deepset-ai/haystack/issues/9828
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
# Step 3: Restart execution after breakpoint with updated snapshot
# ----
# Add the new tool execution decisions to the snapshot and resume execution
new_agent_snapshot = snapshot.agent_snapshot
new_agent_snapshot.tool_execution_decisions = [ToolExecutionDecision.from_dict(ted) for ted in serialized_teds]
result = agent.run(
    # NOTE: Messages are still required but are ignored when passing in a snapshot
    messages=[],
    snapshot=new_agent_snapshot
)
last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")
