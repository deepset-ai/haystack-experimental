# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import create_tool_from_function
from rich.console import Console

from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.components.agents.human_in_the_loop.confirmation_policies import (
    AlwaysAskPolicy,
    AskOncePolicy,
    NeverAskPolicy,
)
from haystack_experimental.components.agents.human_in_the_loop.confirmation_uis import (
    RichConsoleUI,
    SimpleConsoleUI,
)
from haystack_experimental.components.agents.human_in_the_loop.strategies import HumanInTheLoopStrategy


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


def get_phone_number(name: str) -> str:
    """
    Simulate fetching a phone number for a given name.

    :param name: The name of the person.
    :returns:
        A string representing the phone number.
    """
    return f"The phone number for {name} is (123) 456-7890"


phone_tool = create_tool_from_function(
    function=get_phone_number,
    name="get_phone_number",
    description="Get the phone number for a given name.",
)

# Define shared console
cons = Console()

# Define Main Agent with multiple tools and different confirmation strategies
agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[balance_tool, addition_tool, phone_tool],
    system_prompt="You are a helpful financial assistant. Use the provided tool to get bank balances when needed.",
    confirmation_strategies={
        balance_tool.name: HumanInTheLoopStrategy(
            confirmation_policy=AlwaysAskPolicy(), confirmation_ui=RichConsoleUI(console=cons)
        ),
        addition_tool.name: HumanInTheLoopStrategy(
            confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
        ),
        phone_tool.name: HumanInTheLoopStrategy(
            confirmation_policy=AskOncePolicy(), confirmation_ui=SimpleConsoleUI()
        ),
    },
)

# Call bank tool with confirmation (Always Ask) using RichConsoleUI
result = agent.run([ChatMessage.from_user("What's the balance of account 56789?")])
last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")

# Call addition tool with confirmation (Never Ask)
result = agent.run([ChatMessage.from_user("What is 5.5 + 3.2?")])
last_message = result["last_message"]
print(f"\nAgent Result: {last_message.text}")

# Call phone tool with confirmation (Ask Once) using SimpleConsoleUI
result = agent.run([ChatMessage.from_user("What is the phone number of Alice?")])
last_message = result["last_message"]
print(f"\nAgent Result: {last_message.text}")

# Call phone tool again to see that it doesn't ask for confirmation the second time
result = agent.run([ChatMessage.from_user("What is the phone number of Alice?")])
last_message = result["last_message"]
print(f"\nAgent Result: {last_message.text}")
