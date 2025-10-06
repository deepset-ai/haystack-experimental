# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional


class HITLBreakpointException(Exception):
    """
    Exception raised when a tool execution is paused by a ConfirmationStrategy (e.g. BreakpointConfirmationStrategy).
    """

    def __init__(
        self, message: str, tool_name: str, snapshot_file_path: str, tool_call_id: Optional[str] = None
    ) -> None:
        """
        Initialize the HITLBreakpointException.

        :param message: The exception message.
        :param tool_name: The name of the tool whose execution is paused.
        :param snapshot_file_path: The file path to the saved pipeline snapshot.
        :param tool_call_id: Optional unique identifier for the tool call. This can be used to track and correlate
            the decision with a specific tool invocation.
        """
        super().__init__(message)
        self.tool_name = tool_name
        self.snapshot_file_path = snapshot_file_path
        self.tool_call_id = tool_call_id
