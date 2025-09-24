# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ConfirmationUIResult:
    """
    Result of the confirmation UI interaction.

    :param action:
        The action taken by the user such as "confirm", "reject", or "modify".
        This action type is not enforced to allow for custom actions to be implemented.
    :param feedback:
        Optional feedback message from the user. For example, if the user rejects the tool execution,
        they might provide a reason for the rejection.
    :param new_tool_params:
        Optional set of new parameters for the tool. For example, if the user chooses to modify the tool parameters,
        they can provide a new set of parameters here.
    """

    action: str  # "confirm", "reject", "modify"
    feedback: Optional[str] = None
    new_tool_params: Optional[dict[str, Any]] = None


@dataclass
class ToolExecutionDecision:
    """
    Decision made regarding tool execution.

    :param tool_name:
        The name of the tool to be executed.
    :param feedback:
        Optional feedback message if the tool execution was rejected.
    :param final_tool_params:
        Optional final parameters for the tool if execution is confirmed or modified.
    """

    tool_name: str
    feedback: Optional[str] = None
    final_tool_params: Optional[dict[str, Any]] = None
