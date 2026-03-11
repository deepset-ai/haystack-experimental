# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "sandbox_toolset": [
        "E2BSandbox",
        "create_e2b_tools",
        "create_run_bash_command_tool",
        "create_read_file_tool",
        "create_write_file_tool",
        "create_list_directory_tool",
    ]
}

if TYPE_CHECKING:
    from .sandbox_toolset import E2BSandbox as E2BSandbox
    from .sandbox_toolset import create_e2b_tools as create_e2b_tools
    from .sandbox_toolset import create_list_directory_tool as create_list_directory_tool
    from .sandbox_toolset import create_read_file_tool as create_read_file_tool
    from .sandbox_toolset import create_run_bash_command_tool as create_run_bash_command_tool
    from .sandbox_toolset import create_write_file_tool as create_write_file_tool

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
