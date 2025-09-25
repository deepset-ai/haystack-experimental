# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.tools import Tool

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ConfirmationUIResult


class ConfirmationPolicy:
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


class AlwaysAskPolicy(ConfirmationPolicy):
    """Always ask for confirmation."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """
        Always ask for confirmation before executing the tool.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns True, indicating confirmation is needed.
        """
        return True


class NeverAskPolicy(ConfirmationPolicy):
    """Never ask for confirmation."""

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """
        Never ask for confirmation, always proceed with tool execution.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        :returns: Always returns False, indicating no confirmation is needed.
        """
        return False


class AskOncePolicy(ConfirmationPolicy):
    """Ask only once per tool with specific parameters."""

    def __init__(self):
        self._asked_tools = {}

    def should_ask(self, tool: Tool, tool_params: dict[str, Any]) -> bool:
        """
        Ask for confirmation only once per tool with specific parameters.

        :param tool: The tool to be executed.
        :param tool_params: The parameters to be passed to the tool.
        :returns: True if confirmation is needed, False if already asked with the same parameters.
        """
        if tool.name in self._asked_tools and self._asked_tools[tool.name] == tool_params:
            return False
        return True

    def update_after_confirmation(
        self, tool: Tool, tool_params: dict[str, Any], confirmation_result: ConfirmationUIResult
    ) -> None:
        """
        Store the tool and parameters if the action was "confirm" to avoid asking again.

        This method updates the internal state to remember that the user has already confirmed the execution of the
        tool with the given parameters.

        :param tool: The tool that was executed.
        :param tool_params: The parameters that were passed to the tool.
        :param confirmation_result: The result from the confirmation UI.
        """
        if confirmation_result.action == "confirm":
            self._asked_tools[tool.name] = tool_params
