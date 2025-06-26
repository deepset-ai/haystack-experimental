# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "document_length_router": ["DocumentLengthRouter"],
    "document_type_router": ["DocumentTypeRouter"],
}

if TYPE_CHECKING:
    from .document_length_router import DocumentLengthRouter
    from .document_type_router import DocumentTypeRouter
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
