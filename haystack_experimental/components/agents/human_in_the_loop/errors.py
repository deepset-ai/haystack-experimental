# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import AgentBreakpoint


class ToolBreakpointException(BreakpointException):
    """
    Exception raised when a tool execution is paused by a ConfirmationStrategy (e.g. BreakpointConfirmationStrategy).
    """

    def __init__(self, message: str, break_point: AgentBreakpoint):
        """
        Initialize the ToolBreakpointException.

        :param message: The exception message.
        :param break_point: The AgentBreakpoint instance containing the ToolBreakpoint where the tool execution is
            paused.
        """
        super().__init__(message)
        self.break_point = break_point
