# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from rich.console import Console

from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import create_tool_from_function

from haystack_experimental.tools.human_in_the_loop import (
    confirmation_wrapper,
    SimpleInputPrompt,
    RichConsolePrompt,
)


def get_bank_balance(account_id: str) -> str:
    return f"Balance for account {account_id} is $1,234.56"


balance_tool = create_tool_from_function(
    function=get_bank_balance,
    name="get_bank_balance",
    description="Get the bank balance for a given account ID.",
)

#
# Example: Run Tool individually with different Prompts
#

# Use the console version
cons = Console()
console_tool = confirmation_wrapper(balance_tool, RichConsolePrompt(cons))
cons.print("\n[bold]Using console confirmation tool:[/bold]")
res = console_tool.invoke(account_id="123456")
cons.print(f"\n[bold green]Result:[/bold green] {res}")

# Use the simple input version
simple_tool = confirmation_wrapper(balance_tool, SimpleInputPrompt())
print("\nUsing simple input confirmation tool:")
res = simple_tool.invoke(account_id="123456")
print(f"\nResult: {res}")


#
# Example: Running with an Agent
#

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4.1"),
    tools=[console_tool],  # or simple_tool
    system_prompt="""
You are a helpful financial assistant. Use the provided tool to get bank balances when needed.
"""
)

result = agent.run([ChatMessage.from_user("What's the balance of account 56789?")])
last_message = result["last_message"]
cons.print(f"\n[bold green]Agent Result:[/bold green] {last_message.text}")
