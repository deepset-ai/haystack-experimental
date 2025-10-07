# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Optional

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import PipelineSnapshot
from haystack.tools import create_tool_from_function
from rich.console import Console

from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.components.agents.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    BreakpointConfirmationStrategy,
    RichConsoleUI,
    ToolExecutionDecision,
)
from haystack_experimental.components.agents.human_in_the_loop.breakpoint import (
    get_tool_calls_and_descriptions_from_snapshot
)


def get_bank_balance(account_id: str) -> str:
    """
    Simulate fetching a bank balance for a given account ID.

    :param account_id: The ID of the bank account.
    :returns:
        A string representing the bank balance.
    """
    return f"Balance for account {account_id} is $1,234.56"


def addition(a: float, b: float) -> float:
    """
    A simple addition function.

    :param a: First float.
    :param b: Second float.
    :returns:
        Sum of a and b.
    """
    return a + b


def get_latest_snapshot(snapshot_file_path: str) -> PipelineSnapshot:
    """
    Load the latest pipeline snapshot from the 'pipeline_snapshots' directory.
    """
    snapshot_dir = Path(snapshot_file_path)
    possible_snapshots = [snapshot_dir / f for f in os.listdir(snapshot_dir)]
    latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
    return load_pipeline_snapshot(latest_snapshot_file)


def frontend_simulate_tool_decision(
    tool_calls: list[dict[str, Any]], tool_descriptions: dict[str, str], console: Console
) -> list[dict]:
    """
    Simulate front-end receiving tool calls, prompting user, and sending back decisions.

    :param tool_calls:
        A list of tool call dictionaries containing tool_name, id, and arguments.
    :param tool_descriptions:
        A dictionary mapping tool names to their descriptions.
    :param console:
        A Rich Console instance for displaying prompts and messages.
    :returns:
        A list of serialized ToolExecutionDecision dictionaries.
    """

    confirmation_strategy = BlockingConfirmationStrategy(
        confirmation_policy=AlwaysAskPolicy(),
        confirmation_ui=RichConsoleUI(console=console),
    )

    tool_execution_decisions = []
    for tc in tool_calls:
        tool_execution_decisions.append(
            confirmation_strategy.run(
                tool_name=tc["tool_name"],
                tool_description=tool_descriptions[tc["tool_name"]],
                tool_call_id=tc["id"],
                tool_params=tc["arguments"],
            )
        )
    return [ted.to_dict() for ted in tool_execution_decisions]


def run_agent(
    agent: Agent,
    messages: list[ChatMessage],
    console: Console,
    snapshot_file_path: Optional[str] = None,
    tool_execution_decisions: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    """
    Run the agent with the given messages and optional snapshot.
    """
    # Load the latest snapshot if a path is provided
    snapshot = None
    if snapshot_file_path:
        snapshot = get_latest_snapshot(snapshot_file_path=snapshot_file_path)

        # Add any new tool execution decisions to the snapshot
        if tool_execution_decisions:
            teds = [ToolExecutionDecision.from_dict(ted) for ted in tool_execution_decisions]
            existing_decisions = snapshot.agent_snapshot.tool_execution_decisions or []
            snapshot.agent_snapshot.tool_execution_decisions = existing_decisions + teds

    try:
        return agent.run(messages=messages, snapshot=snapshot.agent_snapshot if snapshot else None)
    except BreakpointException as e:
        console.print("[bold red]Execution paused by Breakpoint Confirmation Strategy:[/bold red]", str(e))
        return None


def main(user_message: str):
    """
    Main function to demonstrate the Breakpoint Confirmation Strategy with an agent.
    """
    cons = Console()
    cons.print("\n[bold blue]=== Breakpoint Confirmation Strategy Example ===[/bold blue]\n")
    cons.print(f"[bold yellow]User Message:[/bold yellow] {user_message}\n")

    # Define agent with both tools and breakpoint confirmation strategies
    addition_tool = create_tool_from_function(
        function=addition,
        name="addition",
        description="Add two floats together.",
    )
    balance_tool = create_tool_from_function(
        function=get_bank_balance,
        name="get_bank_balance",
        description="Get the bank balance for a given account ID.",
    )
    snapshot_fp = "pipeline_snapshots"
    bank_agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
        tools=[balance_tool, addition_tool],
        system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
        confirmation_strategies={
            balance_tool.name: BreakpointConfirmationStrategy(snapshot_file_path=snapshot_fp),
            addition_tool.name: BreakpointConfirmationStrategy(snapshot_file_path=snapshot_fp),
        },
    )

    # Step 1: Initial run
    result = run_agent(bank_agent, [ChatMessage.from_user(user_message)], cons)

    # Step 2: Loop to handle break point confirmation strategy until agent completes
    while result is None:
        # Load the latest snapshot from disk and prep data for front-end
        loaded_snapshot = get_latest_snapshot(snapshot_file_path=snapshot_fp)
        serialized_tool_calls, tool_descripts = get_tool_calls_and_descriptions_from_snapshot(
            agent_snapshot=loaded_snapshot.agent_snapshot, breakpoint_tool_only=True
        )

        # Simulate front-end interaction
        serialized_teds = frontend_simulate_tool_decision(serialized_tool_calls, tool_descripts, cons)

        # Re-run the agent with the new tool execution decisions
        result = run_agent(bank_agent, [], cons, snapshot_fp, serialized_teds)

    # Step 3: Final result
    last_message = result["last_message"]
    cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")


if __name__ == "__main__":
    for usr_msg in [
        # Single tool call question --> Works
        "What's the balance of account 56789?",
        # Two tool call question --> Works
        "What's the balance of account 56789 and what is 5.5 + 3.2?",
        # Multiple sequential tool calls question --> Works
        "What's the balance of account 56789? If it's lower than $2000, what's the balance of account 12345?",
    ]:
        main(usr_msg)
