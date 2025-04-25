import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "file_to_image": ["ImageFileToImageContent"],
}

if TYPE_CHECKING:
    from .file_to_image import ImageFileToImageContent
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
