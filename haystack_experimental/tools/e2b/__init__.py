# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "e2b_sandbox": ["E2BSandbox"],
    "bash_tool": ["RunBashCommandTool"],
    "read_file_tool": ["ReadFileTool"],
    "write_file_tool": ["WriteFileTool"],
    "list_directory_tool": ["ListDirectoryTool"],
    "sandbox_toolset": ["E2BToolset"],
}

if TYPE_CHECKING:
    from .bash_tool import RunBashCommandTool as RunBashCommandTool
    from .e2b_sandbox import E2BSandbox as E2BSandbox
    from .list_directory_tool import ListDirectoryTool as ListDirectoryTool
    from .read_file_tool import ReadFileTool as ReadFileTool
    from .sandbox_toolset import E2BToolset as E2BToolset
    from .write_file_tool import WriteFileTool as WriteFileTool

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
