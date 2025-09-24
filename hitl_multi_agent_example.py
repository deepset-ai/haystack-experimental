# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import create_tool_from_function, ComponentTool
from rich.console import Console

from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.tools.hitl import (
    AlwaysAskPolicy,
    HumanInTheLoopStrategy,
    RichConsoleUI,
)


def addition(a: float, b: float) -> float:
    """
    A simple addition function.

    :param a: First float.
    :param b: Second float.
    :returns:
        Sum of a and b.
    """
    return a + b


addition_tool = create_tool_from_function(
    function=addition,
    name="addition",
    description="Add two floats together.",
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

# Define shared console for all UIs
cons = Console()

# Define Bank Sub-Agent
bank_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[balance_tool],
    system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
    confirmation_strategies={
        balance_tool.name: HumanInTheLoopStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI(console=cons)
        ),
    },
)
bank_agent_tool = ComponentTool(
    component=bank_agent,
    name="bank_agent_tool",
    description="A bank agent that can get bank balances.",
)

# Define Math Sub-Agent
math_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[addition_tool],
    system_prompt="You are a helpful math assistant. Use the provided tool to perform addition when needed.",
    confirmation_strategies={
        addition_tool.name: HumanInTheLoopStrategy(
            # We use AlwaysAskPolicy here for demonstration; in real scenarios, you might choose NeverAskPolicy
            confirmation_policy=AlwaysAskPolicy(),
            confirmation_ui=RichConsoleUI(console=cons)
        ),
    },
)
math_agent_tool = ComponentTool(
    component=math_agent,
    name="math_agent_tool",
    description="A math agent that can perform addition.",
)

# Define Main Agent with Sub-Agents as tools
planner_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[bank_agent_tool, math_agent_tool],
    system_prompt="""You are a master agent that can delegate tasks to sub-agents based on the user's request.
Available sub-agents:
- bank_agent_tool: A bank agent that can get bank balances.
- math_agent_tool: A math agent that can perform addition.
Use the appropriate sub-agent to handle the user's request.
""",
)

# Make bank balance request to planner agent
result = planner_agent.run([ChatMessage.from_user("What's the balance of account 56789?")])
last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")

# Make addition request to planner agent
result = planner_agent.run([ChatMessage.from_user("What is 5.5 + 3.2?")])
last_message = result["last_message"]
print(f"\nAgent Result: {last_message.text}")

# Make bank balance request and addition request to planner agent
result = planner_agent.run([ChatMessage.from_user("What's the balance of account 56789 and what is 5.5 + 3.2?")])
last_message = result["last_message"]
print(f"\nAgent Result: {last_message.text}")
