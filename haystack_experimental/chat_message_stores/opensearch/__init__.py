# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {"chat_message_store": ["OpenSearchChatMessageStore"]}

if TYPE_CHECKING:
    from .chat_message_store import OpenSearchChatMessageStore as OpenSearchChatMessageStore
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
