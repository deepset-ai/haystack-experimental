# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "dataclasses": ["ToolExecutionDecision"],
    "errors": ["HITLBreakpointException"],
    "strategies": ["BreakpointConfirmationStrategy"],
}

if TYPE_CHECKING:
    from .dataclasses import ToolExecutionDecision as ToolExecutionDecision
    from .errors import HITLBreakpointException as HITLBreakpointException
    from .strategies import BreakpointConfirmationStrategy as BreakpointConfirmationStrategy

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
