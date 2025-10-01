# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from haystack.core.errors import BreakpointException
from haystack.components.generators.chat import OpenAIChatGenerator
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


def get_latest_snapshot():
    snapshot_dir = Path("pipeline_snapshots")
    possible_snapshots = [snapshot_dir / f for f in os.listdir(snapshot_dir)]
    latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
    return load_pipeline_snapshot(latest_snapshot_file)


def frontend_simulate_tool_execution(agent_snapshot, console):
    """Simulate front-end receiving tool calls, prompting user, and sending back decisions."""
    serialized_tool_calls, tool_descriptions = get_tool_calls_and_descriptions(agent_snapshot)

    confirmation_strategy = HumanInTheLoopStrategy(
        confirmation_policy=AlwaysAskPolicy(),
        confirmation_ui=RichConsoleUI(console=console),
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
    return [ted.to_dict() for ted in tool_execution_decisions]


def run_agent(agent, messages, console, snapshot=None):
    try:
        return agent.run(messages=messages, snapshot=snapshot.agent_snapshot if snapshot else None)
    except BreakpointException as e:
        console.print(
            "[bold red]Execution paused by Breakpoint Confirmation Strategy:[/bold red]",
            str(e)
        )
        return None


if __name__ == "__main__":
    cons = Console()
    cons.print("\n[bold blue]=== Multiple Sequential Tool Calls Example ===[/bold blue]\n")

    balance_tool = create_tool_from_function(
        function=get_bank_balance,
        name="get_bank_balance",
        description="Get the bank balance for a given account ID.",
    )
    bank_agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
        tools=[balance_tool],
        system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
        confirmation_strategies={balance_tool.name: BreakpointConfirmationStrategy()}
    )

    # Step 1: Initial run
    user_message = "What's the balance of account 56789?"
    # user_message = "What's the balance of account 56789? If it's lower than $2000, what's the balance of account 12345?"
    result = run_agent(bank_agent, [ChatMessage.from_user(user_message)], cons)

    # Step 2 & 4: Loop to handle breakpoints until agent completes
    while result is None:
        loaded_snapshot = get_latest_snapshot()
        serialized_teds = frontend_simulate_tool_execution(loaded_snapshot.agent_snapshot, cons)
        loaded_snapshot.agent_snapshot.tool_execution_decisions = [
            ToolExecutionDecision.from_dict(ted) for ted in serialized_teds
        ]
        result = run_agent(bank_agent, [], cons, loaded_snapshot)

    # Step 5: Final result
    last_message = result["last_message"]
    cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")
