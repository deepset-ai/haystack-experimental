# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from haystack.tools import Tool
    from haystack_experimental.tools.human_in_the_loop import ConfirmationResult


class ConfirmationPrompt(Protocol):
    def confirm(self, tool_name: str, params: dict[str, Any]) -> ConfirmationResult:
        """
        Ask for user confirmation before executing a tool.

        :param tool_name: Name of the tool to be executed.
        :param params: Parameters to be passed to the tool.
        :returns:
            ConfirmationResult with action (e.g. "confirm" or "reject") and optional feedback message.
        """


class ExecutionPolicy(Protocol):
    def handle(self, result: ConfirmationResult, tool: Tool, kwargs: dict[str, Any]) -> Any:
        ...
