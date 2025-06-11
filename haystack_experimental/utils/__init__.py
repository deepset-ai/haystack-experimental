# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "base_serialization": ["_serialize_value_with_schema", "_deserialize_value_with_schema"],
}

if TYPE_CHECKING:
    from .base_serialization import (
        _deserialize_value_with_schema,
        _serialize_value_with_schema,
    )
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
