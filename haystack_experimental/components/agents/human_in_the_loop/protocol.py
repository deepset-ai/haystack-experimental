# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Protocol

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ToolExecutionDecision

if TYPE_CHECKING:
    from haystack.tools import Tool


class ConfirmationStrategy(Protocol):
    def run(self, tool: "Tool", tool_params: dict[str, Any]) -> ToolExecutionDecision:
        """
        Run the confirmation strategy for a given tool and its parameters.

        :param tool: The tool to be confirmed.
        :param tool_params: The parameters to be passed to the tool.

        :returns:
            The result of the confirmation strategy (e.g., tool output, rejection message, etc.).
        """

    def to_dict(self) -> dict[str, Any]:
        """Serialize the strategy to a dictionary."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationStrategy":
        """Deserialize the strategy from a dictionary."""
