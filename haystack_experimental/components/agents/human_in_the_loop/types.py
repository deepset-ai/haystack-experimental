# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Protocol

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.tools import Tool

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import (
    ConfirmationUIResult,
    ToolExecutionDecision,
)


class ConfirmationUI(Protocol):
    """Base class for confirmation UIs."""

    def get_user_confirmation(self, tool: Tool, tool_params: dict[str, Any]) -> ConfirmationUIResult:
        """Get user confirmation for tool execution."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize the UI to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationUI":
        """Deserialize the ConfirmationUI from a dictionary."""
        return default_from_dict(cls, data)


class ConfirmationPolicy(Protocol):
    """Base class for confirmation policies."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """Determine whether to ask for confirmation."""
        raise NotImplementedError

    def update_after_confirmation(
        self, tool: Tool, tool_params: dict[str, Any], confirmation_result: ConfirmationUIResult
    ) -> None:
        """Update the policy based on the confirmation UI result."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize the policy to a dictionary."""
        return default_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConfirmationPolicy":
        """Deserialize the policy from a dictionary."""
        return default_from_dict(cls, data)


class ConfirmationStrategy(Protocol):
    def run(self, tool: Tool, tool_params: dict[str, Any]) -> ToolExecutionDecision:
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
