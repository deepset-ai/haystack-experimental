# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import component, logging
from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream
from haystack.utils import expand_page_range

from haystack_experimental.components.converters.image_utils import DETAIL_TO_IMAGE_SIZE, read_image_from_pdf
from haystack_experimental.dataclasses.chat_message import ImageContent

logger = logging.getLogger(__name__)


@component
class PDFToImageContent:
    """
    Converts PDF files to ImageContent objects.
    """

    def __init__(
        self,
        *,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        downsize: bool = False,
        page_range: Optional[List[Union[str, int]]] = None,
    ):
        """
        Create the PDFToImageContent component.

        :param detail: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
        :param downsize: If True, the image will be downscaled to the specified detail level.
        :param page_range: A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
            metadata from the first and third pages of each document. It also accepts printable range strings, e.g.:
            ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,11, 12.
            If None, metadata will be extracted from the entire document for each document in the documents list.
        """
        self.detail = detail
        self.downsize = downsize
        self.page_range = page_range

    @component.output_types(image_contents=List[ImageContent])
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        *,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        downsize: Optional[bool] = None,
        page_range: Optional[List[Union[str, int]]] = None,
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
        :param downsize:
            If True, the image will be downscaled to the specified detail level.
            If not provided, the downsize value will be the one set in the constructor.
        :param page_range:
            A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
            metadata from the first and third pages of each document. It also accepts printable range strings, e.g.:
            ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,11, 12.
            If None, metadata will be extracted from the entire document for each document in the documents list.
            If not provided, the page_range value will be the one set in the constructor.

        :returns:
            A dictionary with the following keys:
            - `image_contents`: A list of ImageContent objects.
        """
        if not sources:
            return {"image_contents": []}

        detail = detail or self.detail
        downsize = downsize or self.downsize
        page_range = page_range or self.page_range

        size = DETAIL_TO_IMAGE_SIZE[detail] if detail else None
        expanded_page_range = expand_page_range(page_range) if page_range else None

        image_contents = []

        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            if isinstance(source, str):
                source = Path(source)

            mime_type = None
            if isinstance(source, Path):
                mime_type = mimetypes.guess_type(source.as_posix())[0]
            elif isinstance(source, ByteStream):
                mime_type = source.mime_type

            try:
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning("Could not read {source}. Skipping it. Error: {error}", source=source, error=e)
                continue
            try:
                # we need base64 here
                # TODO Should add the page number to the metadata
                #      Update function to return additional metadata
                base64_images = read_image_from_pdf(
                    bytestream=bytestream, page_range=expanded_page_range, size=size, downsize=downsize
                )
            except Exception as e:
                logger.warning(
                    "Could not convert file {source}. Skipping it. Error message: {error}", source=source, error=e,
                )
                continue

            # TODO Add additional metadata here
            merged_metadata = {**bytestream.meta, **metadata}

            image_contents.extend(
                [
                    ImageContent(base64_image=image, mime_type=mime_type, meta=merged_metadata, detail=detail)
                    for image in base64_images
                ]
            )

        return {"image_contents": image_contents}
