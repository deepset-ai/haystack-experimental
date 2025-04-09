# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    pass


class PipelineConnectError(PipelineError):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelineDrawingError(PipelineError):
    pass


class PipelineMaxComponentRuns(PipelineError):
    pass


class PipelineUnmarshalError(PipelineError):
    pass


class ComponentError(Exception):
    pass


class ComponentDeserializationError(Exception):
    pass


class DeserializationError(Exception):
    pass


class SerializationError(Exception):
    pass


class PipelineBreakpointException(Exception):
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
