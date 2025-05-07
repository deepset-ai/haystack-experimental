# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import filetype
from haystack import logging
from haystack.components.fetchers.link_content import LinkContentFetcher

from haystack_experimental.components.image_converters.image_utils import MIME_TO_FORMAT

logger = logging.getLogger(__name__)

IMAGE_MIME_TYPES = {key for key in MIME_TO_FORMAT.keys() if key != "application/pdf"}


@dataclass
class ImageContent:
    """
    The image content of a chat message.

    :param base64_image: A base64 string representing the image.
    :param mime_type: The MIME type of the image (e.g. "image/png", "image/jpeg").
        Providing this value is recommended, as most LLM providers require it.
        If not provided, the MIME type is guessed from the base64 string, which can be slow and not always reliable.
    :param detail: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
    :param meta: Optional metadata for the image.
    """

    base64_image: str
    mime_type: Optional[str] = None
    detail: Optional[Literal["auto", "high", "low"]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # mime_type is an important information, so we try to guess it if not provided
        if not self.mime_type:
            try:
                # Attempt to decode the string as base64
                decoded_image = base64.b64decode(self.base64_image)

                guess = filetype.guess(decoded_image)
                if guess:
                    self.mime_type = guess.mime
                else:
                    msg = (
                        "Failed to guess the MIME type of the image. Omitting the MIME type may result in "
                        "processing errors or incorrect handling of the image by LLM providers."
                    )
                    logger.warning(msg)
            except:
                pass

    def __repr__(self) -> str:
        """
        Return a string representation of the ImageContent, truncating the base64_image to 100 bytes.
        """
        fields = []

        truncated_data = self.base64_image[:100] + "..." if len(self.base64_image) > 100 else self.base64_image
        fields.append(f"base64_image={truncated_data!r}")
        fields.append(f"mime_type={self.mime_type!r}")
        fields.append(f"detail={self.detail!r}")
        fields.append(f"meta={self.meta!r}")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}({fields_str})"

    @classmethod
    def from_file_path(
        cls,
        file_path: Union[str, Path],
        *,
        size: Optional[Tuple[int, int]] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ImageContent":
        """
        Create an ImageContent object from a file path.

        It exposes similar functionality as the `ImageFileToImageContent` component. For PDF to ImageContent conversion,
        use the `PDFToImageContent` component.

        :param file_path:
            The path to the image file. PDF files are not supported. For PDF to ImageContent conversion, use the
            `PDFToImageContent` component.
        :param size:
            If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param detail:
            Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
        :param meta:
            Additional metadata for the image.

        :returns:
            An ImageContent object.
        """
        # to avoid a circular import
        from haystack_experimental.components.image_converters import ImageFileToImageContent

        converter = ImageFileToImageContent(size=size, detail=detail)
        result = converter.run(sources=[file_path], meta=[meta] if meta else None)
        return result["image_contents"][0]


    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        retry_attempts: int = 2,
        timeout: int = 10,
        size: Optional[Tuple[int, int]] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "ImageContent":
        """
        Create an ImageContent object from a URL. The image is downloaded and converted to a base64 string.

        For PDF to ImageContent conversion, use the `PDFToImageContent` component.

        :param url:
            The URL of the image. PDF files are not supported. For PDF to ImageContent conversion, use the
            `PDFToImageContent` component.
        :param retry_attempts:
            The number of times to retry to fetch the URL's content.
        :param timeout:
            Timeout in seconds for the request.
        :param size:
            If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        :param detail:
            Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
        :param meta:
            Additional metadata for the image.

        :raises ValueError:
            If the URL does not point to an image.

        :returns:
            An ImageContent object.
        """
        # to avoid a circular import
        from haystack_experimental.components.image_converters import ImageFileToImageContent

        fetcher = LinkContentFetcher(raise_on_failure=True, retry_attempts=retry_attempts, timeout=timeout)
        bytestream = fetcher.run(urls=[url])["streams"][0]

        if bytestream.mime_type not in IMAGE_MIME_TYPES:
            msg = f"The URL does not point to an image. The MIME type of the URL is {bytestream.mime_type}."
            raise ValueError(msg)

        converter = ImageFileToImageContent(size=size, detail=detail)
        result = converter.run(sources=[bytestream], meta=[meta] if meta else None)
        return result["image_contents"][0]
