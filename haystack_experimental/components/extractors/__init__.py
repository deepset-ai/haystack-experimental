# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "llm_document_content_extractor": ["LLMDocumentContentExtractor"],
}

if TYPE_CHECKING:
    from .llm_document_content_extractor import LLMDocumentContentExtractor as LLMDocumentContentExtractor

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
