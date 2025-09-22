# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from haystack.tools import Tool

    from haystack_experimental.tools.hitl import ConfirmationUIResult, ToolExecutionDecision


class ConfirmationPolicy(Protocol):
    """Protocol for confirmation policies."""

    def should_ask(self, tool: "Tool", tool_params: dict[str, Any]) -> bool:
        """Determine whether to ask for confirmation."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a dictionary."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy":
        """Deserialize the policy from a dictionary."""


class ConfirmationUI(Protocol):
    """Protocol for confirmation user interfaces."""

    def get_user_confirmation(self, tool: "Tool", tool_params: dict[str, Any]) -> "ConfirmationUIResult":
        """Get user confirmation for tool execution."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the UI to a dictionary."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationUI":
        """Deserialize the UI from a dictionary."""


class ConfirmationStrategy(Protocol):
    def run(self, tool: "Tool", tool_params: dict[str, Any]) -> "ToolExecutionDecision":
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
