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
    )
)
messages = [ChatMessage.from_user("What's the balance of account 56789?")]
try:
    _ = agent.run(
        messages=messages,
        break_point=agent_break_point
    )
except BreakpointException:
    cons.print("[bold red]Execution paused at breakpoint.[/bold red]")

# NOTE: Opened issue to make it easier to find the latest snapshot: https://github.com/deepset-ai/haystack/issues/9828
possible_snapshots = [Path("pipeline_snapshots") / f for f in os.listdir(Path("pipeline_snapshots"))]
latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
snapshot = load_pipeline_snapshot(latest_snapshot_file)

# TODO Ask for user feedback to modify tool parameters

# Restart execution after breakpoint
# NOTE: We extended the Agent in haystack-experimental to allow running with a snapshot and a breakpoint
result = agent.run(
    # TODO Messages are still required but are ignored when passing in a snapshot
    messages=messages,
    snapshot=snapshot.agent_snapshot,
    # This break point shouldn't trigger again, since the agent will exit after the next llm call to give the
    # final answer.
    break_point=replace(
        snapshot.agent_snapshot.break_point,
        break_point=replace(
            snapshot.agent_snapshot.break_point.break_point,
            visit_count=snapshot.agent_snapshot.break_point.break_point.visit_count + 1
        )
    ),
)
last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")


# ============
# Using Only Breakpoint Feature: Multiple sequential Tool Calls
# ============
cons.print("\n[bold blue]=== Multiple Sequential Tool Calls Example ===[/bold blue]\n")
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
        component_name="tool_invoker",
        visit_count=0,
        snapshot_file_path="pipeline_snapshots",
    )
)
messages = [
    ChatMessage.from_user(
        # We create a scenario where the agent needs to call the tool multiple times in a single run sequentially
        "What's the balance of account 56789? If it's lower than $2000, what's the balance of account 12345?"
    )
]
try:
    _ = agent.run(
        messages=messages,
        break_point=agent_break_point
    )
except BreakpointException:
    cons.print("[bold red]Execution paused at breakpoint.[/bold red]")

# Load the snapshot
possible_snapshots = [Path("pipeline_snapshots") / f for f in os.listdir(Path("pipeline_snapshots"))]
latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
snapshot = load_pipeline_snapshot(latest_snapshot_file)

# TODO Ask for user feedback to modify tool parameters

# Restart execution after breakpoint
# NOTE: We extended the Agent in haystack-experimental to allow running with a snapshot and a breakpoint
try:
    _ = agent.run(
        messages=messages,
        snapshot=snapshot.agent_snapshot,
        # This break point should trigger again b/c the agent needs to call the tool a second time since the bank account
        # balance for account 56789 is below $2000
        break_point=replace(
            snapshot.agent_snapshot.break_point,
            break_point=replace(
                snapshot.agent_snapshot.break_point.break_point,
                visit_count=snapshot.agent_snapshot.break_point.break_point.visit_count + 1
            )
        ),
    )
except BreakpointException:
    cons.print("[bold red]Execution paused at breakpoint (2nd time).[/bold red]")

# Load the snapshot
possible_snapshots = [Path("pipeline_snapshots") / f for f in os.listdir(Path("pipeline_snapshots"))]
latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
snapshot = load_pipeline_snapshot(latest_snapshot_file)

# Restart execution after breakpoint
result = agent.run(
    messages=messages,
    snapshot=snapshot.agent_snapshot,
    # Increment the visit count so break point only triggers if there is a 3rd tool call (which there shouldn't be)
    break_point=replace(
        snapshot.agent_snapshot.break_point,
        break_point=replace(
            snapshot.agent_snapshot.break_point.break_point,
            visit_count=snapshot.agent_snapshot.break_point.break_point.visit_count + 1
        )
    ),
)
last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")


# ============
# Combining breakpoint and confirmation strategies
# ============
# cons.print("\n[bold blue]=== Combining Breakpoint and Confirmation Strategies ===[/bold blue]\n")

# Is it possible to follow the above pattern but use a confirmation strategy instead to trigger the breakpoint?

# - Probably necessitates a new ConfirmationStrategy that is capable of triggering a breakpoint and then also resuming
# - Could optionally still use the existing UI to get user feedback on modifying tool parameters before triggering the
#   break point. This would have to be optional b/c the UIs are not suitable for a backend service.

# agent = Agent(
#     chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
#     tools=[balance_tool],
#     system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
#     confirmation_strategies={
#         balance_tool.name: HumanInTheLoopStrategy(
#             confirmation_policy=AlwaysAskPolicy(),
#             confirmation_ui=RichConsoleUI(console=cons)
#         ),
#     },
# )
