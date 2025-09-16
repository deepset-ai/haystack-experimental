# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, replace
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.tools.types.protocol import ConfirmationPrompt, ExecutionPolicy


@dataclass
class ConfirmationResult:
    action: str   # "confirm" | "reject" | "modify"
    feedback: str | None = None
    new_params: dict[str, Any] | None = None


# Different prompt
class RichConsolePrompt:
    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def confirm(self, tool_name: str, params: dict[str, Any]) -> ConfirmationResult:
        # Display info
        lines = [f"[bold yellow]Tool:[/bold yellow] {tool_name}"]
        if params:
            lines.append("\n[bold yellow]Arguments:[/bold yellow]")
            for k, v in params.items():
                lines.append(f"\n[cyan]{k}:[/cyan]\n  {v}")
        self.console.print(Panel("\n".join(lines), title="🔧 Tool Execution Request"))

        # Ask action
        choice = Prompt.ask(
            "\nYour choice",
            choices=["y", "n", "m"],  # confirm, reject, modify
            default="y",
        )
        if choice == "y":
            return ConfirmationResult(action="confirm")
        elif choice == "m":
            new_params = {}
            for k, v in params.items():
                new_val = Prompt.ask(f"Modify '{k}'", default=str(v))
                new_params[k] = new_val
            return ConfirmationResult(action="modify", new_params=new_params)
        else:  # reject
            feedback = Prompt.ask("Feedback message (optional)", default="")
            return ConfirmationResult(action="reject", feedback=feedback or None)


class SimpleInputPrompt:
    def confirm(self, tool_name: str, params: dict[str, Any]) -> ConfirmationResult:
        print(f"Tool: {tool_name}")
        if params:
            print("Arguments:")
            for k, v in params.items():
                print(f"  {k}: {v}")

        choice = input("Confirm execution? (y=confirm / n=reject / m=modify): ").strip().lower()
        if choice == "y":
            return ConfirmationResult(action="confirm")
        elif choice == "m":
            new_params = {}
            for k, v in params.items():
                new_val = input(f"Modify '{k}' [{v}]: ").strip() or v
                new_params[k] = new_val
            return ConfirmationResult(action="modify", new_params=new_params)
        else:  # modify
            feedback = input("Feedback message (optional): ").strip()
            return ConfirmationResult(action="reject", feedback=feedback or None)


class DefaultPolicy:
    def handle(self, result: ConfirmationResult, tool: Tool, kwargs: dict[str, Any]) -> Any:
        if result.action == "reject":
            return {
                "status": "rejected",
                "tool": tool.name,
                "feedback": result.feedback or "Tool execution rejected by user",
            }
        elif result.action == "modify" and result.new_params:
            # Run immediately with new params
            return tool.function(**result.new_params)
        return tool.function(**kwargs)


class AutoConfirmPolicy:
    def handle(self, result: ConfirmationResult, tool: Tool, kwargs: dict[str, Any]) -> Any:
        # Always run, ignore user rejection
        return tool.function(**kwargs)


# Wrapper
def confirmation_wrapper(
    tool: Tool,
    strategy: ConfirmationPrompt,
    policy: ExecutionPolicy = DefaultPolicy(),
) -> Tool:
    def wrapped_function(**kwargs: Any) -> Any:
        result = strategy.confirm(tool.name, kwargs)
        return policy.handle(result, tool, kwargs)
    return replace(tool, function=wrapped_function)


# Main
if __name__ == "__main__":
    def get_bank_balance(account_id: str) -> str:
        return f"Balance for account {account_id} is $1,234.56"

    balance_tool = create_tool_from_function(
        function=get_bank_balance,
        name="get_bank_balance",
        description="Get the bank balance for a given account ID.",
    )

    cons = Console()
    console_tool = confirmation_wrapper(balance_tool, RichConsolePrompt(cons))
    simple_tool = confirmation_wrapper(balance_tool, SimpleInputPrompt())

    # Use the console version
    cons.print("\n[bold]Using console confirmation tool:[/bold]")
    res = console_tool.invoke(account_id="123456")
    cons.print(f"\n[bold green]Result:[/bold green] {res}")

    # Use the simple input version
    print("\nUsing simple input confirmation tool:")
    res = simple_tool.invoke(account_id="123456")
    print(f"\nResult: {res}")
