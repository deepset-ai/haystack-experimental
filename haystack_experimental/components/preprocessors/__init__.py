# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "embedding_based_document_splitter": ["EmbeddingBasedDocumentSplitter"],
    "md_header_level_inferrer": ["MarkdownHeaderLevelInferrer"],
}

if TYPE_CHECKING:
    from .embedding_based_document_splitter import EmbeddingBasedDocumentSplitter
    from .md_header_level_inferrer import MarkdownHeaderLevelInferrer

else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
