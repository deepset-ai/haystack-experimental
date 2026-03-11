# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "sandbox_toolset": [
        "E2BSandbox",
        "RunBashCommandTool",
        "ReadFileTool",
        "WriteFileTool",
        "ListDirectoryTool",
        "create_e2b_tools",
    ]
}

if TYPE_CHECKING:
    from .sandbox_toolset import E2BSandbox as E2BSandbox
    from .sandbox_toolset import ListDirectoryTool as ListDirectoryTool
    from .sandbox_toolset import ReadFileTool as ReadFileTool
    from .sandbox_toolset import RunBashCommandTool as RunBashCommandTool
    from .sandbox_toolset import WriteFileTool as WriteFileTool
    from .sandbox_toolset import create_e2b_tools as create_e2b_tools

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
