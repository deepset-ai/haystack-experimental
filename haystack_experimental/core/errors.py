# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional


class BreakpointException(Exception):
    """
    Exception raised when a pipeline breakpoint is triggered.
    """

    def __init__(
        self,
        message: str,
        component: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.component = component
        self.state = state
        self.results = results


class PipelineInvalidResumeStateError(Exception):
    """
    Exception raised when a pipeline is resumed from an invalid state.
    """

    def __init__(self, message: str):
        super().__init__(message)
