# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "chat_message": ["ChatMessage"],
    "image_content": ["ImageContent"],
    "answer": ["GeneratedAnswer"],
}

if TYPE_CHECKING:
    from .answer import GeneratedAnswer
    from .chat_message import ChatMessage
    from .image_content import ImageContent
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
