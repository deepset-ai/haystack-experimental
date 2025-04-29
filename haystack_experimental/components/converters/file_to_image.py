# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

from haystack_experimental.components.converters.image_utils import (
    DETAIL_TO_IMAGE_SIZE,
    encode_image_to_base64,
)
from haystack_experimental.dataclasses.chat_message import ImageContent

with LazyImport(
    "Both 'detail' and 'downsize' parameters are set. "
    "Image resizing will be applied, which requires the Pillow library. "
    "Run 'pip install pillow'"
) as pillow_import:
    import PIL  # pylint: disable=unused-import

logger = logging.getLogger(__name__)


_EMPTY_BYTE_STRING = b""


@component
class ImageFileToImageContent:
    """
    Converts image files to ImageContent objects.
    """

    def __init__(self, *, detail: Optional[Literal["auto", "high", "low"]] = None, downsize: bool = False):
        """
        Create the ImageFileToImageContent component.

        :param detail: Determines the target size of the converted images. Maps to predefined dimensions:
            - "auto": Uses dimensions (768, 2048)
            - "high": Uses dimensions (768, 2048)
            - "low": Uses dimensions (512, 512)
            The dimensions are specified as (width, height) tuples.
            When combined with `downsize=True`, these dimensions serve as maximum constraints while maintaining the
            original aspect ratio.
        :param downsize: If True, resizes the image to fit within the specified dimensions while maintaining aspect
            ratio. This reduces file size, memory usage, and processing time, which is beneficial when working with
            models that have resolution constraints or when transmitting images to remote services.
        """
        self.detail = detail
        self.downsize = downsize

        if self.detail and self.downsize:
            pillow_import.check()

    @component.output_types(image_contents=List[ImageContent])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        downsize: Optional[bool] = None,
    ):
        """
        Converts files to ImageContent objects.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources as they're zipped together.
            For ByteStream objects, their `meta` is added to the output documents.
        :param detail:
            The detail level of the image content.
            If not provided, the detail level will be the one set in the constructor.
        :param downsize: If True, resizes the image to fit within the specified dimensions while maintaining aspect
            ratio. This reduces file size, memory usage, and processing time, which is beneficial when working with
            models that have resolution constraints or when transmitting images to remote services.
            If not provided, the downsize value will be the one set in the constructor.

        :returns:
            A dictionary with the following keys:
            - `image_contents`: A list of ImageContent objects.
        """
        if not sources:
            return {"image_contents": []}

        detail = detail or self.detail
        downsize = downsize or self.downsize
        size = DETAIL_TO_IMAGE_SIZE[detail] if detail else None

        image_contents = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            if isinstance(source, str):
                source = Path(source)

            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue

            if bytestream.mime_type is None and isinstance(source, Path):
                bytestream.mime_type = mimetypes.guess_type(source.as_posix())[0]

            if bytestream.data == _EMPTY_BYTE_STRING:
                logger.warning("File {source} is empty. Skipping it.", source=source)
                continue

            try:
                inferred_mime_type, base64_image = encode_image_to_base64(
                    bytestream=bytestream, size=size, downsize=downsize
                )
            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e
                )
                continue

            merged_metadata = {**bytestream.meta, **metadata}
            image_content = ImageContent(
                base64_image=base64_image, mime_type=inferred_mime_type, meta=merged_metadata, detail=detail
            )
            image_contents.append(image_content)

        return {"image_contents": image_contents}
