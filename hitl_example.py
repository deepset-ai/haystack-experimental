from rich.console import Console

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
