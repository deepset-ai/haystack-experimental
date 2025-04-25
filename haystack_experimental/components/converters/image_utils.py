# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack import logging
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pypdf pdf2image'") as pypdf_and_pdf2image_import:
    from pypdf import PdfReader
    import pdf2image

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image as PILImage
    from PIL.Image import Image


logger = logging.getLogger(__name__)


DETAIL_TO_IMAGE_SIZE = {"low": (512, 512), "high": (768, 2_048), "auto": (768, 2_048)}


def open_image_to_base64(
    fp: Union[Path, bytes], size: Union[Tuple[int, int], Dict[float, Tuple[int, int]]], mime_type: Optional[str] = None,
) -> str:
    """
    Open an image from a file path.

    :param fp: A filename (string), os.PathLike object or a file object.
       The file object must implement ``file.read``, ``file.seek``, and ``file.tell`` methods, and be opened in
       binary mode.
    :param size: Tuple of (num_pixels, num_pixels) or a dictionary with aspect ratios as keys and tuples of
        (num_pixels, num_pixels) as values
    :param mime_type: The mime type to load and save the image in. If not provided, it will be guessed from the
        file name.
    :return: Base64 string of the image
    """
    pillow_import.check()
    formats = [mime_type] if mime_type else None
    image = PILImage.open(fp, formats=formats)
    image = downsize_image(image, size)
    return to_base64_str(image, mime_type=mime_type)


def to_base64_str(image: "Image", mime_type: str = "image/jpeg") -> str:
    """
    Convert PIL Image to base64 string based on the specified mime type.

    :param image: PIL Image object
    :param mime_type: The mime type to save the image in. Default is "image/jpeg".
    :return: Base64 string of the image
    """
    # Convert image to RGB if it has an alpha channel and we are saving as JPEG
    if (
        mime_type == "image/jpeg" or mime_type == "image/jpg"
    ) and (
        image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    ):
        image = image.convert("RGB")

    buffered = BytesIO()
    image.save(buffered, format=mime_type)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def downsize_image(image: "Image", size: Union[Tuple[int, int], Dict[float, Tuple[int, int]]]) -> "Image":
    """
    Resize the image to be smaller while maintaining the aspect ratio.

    The latency of vision models can be improved by downsizing images ahead of time to be less than the
    maximum size they are expected to be. You can specify the size as a tuple of (num_pixels, num_pixels) or
    as a dictionary with tuples of (num_pixels, num_pixels) as values and aspect ratios as keys. The value of the
    key being closest to the image aspect ratio is used.

    :param image: PIL Image object
    :param size: Tuple of (num_pixels, num_pixels) or a dictionary with aspect ratios as keys and tuples of
        (num_pixels, num_pixels) as values
    :return: Resized PIL Image object
    """
    longer_dim = max(image.size)
    shorter_dim = min(image.size)
    aspect_ratio = shorter_dim / longer_dim

    if isinstance(size, dict):
        nearest_aspect_ratio_key = min(size, key=lambda aspect_ratio_key: abs(aspect_ratio - aspect_ratio_key))
        size = size[nearest_aspect_ratio_key]

    max_longer_dim = max(size)
    max_shorter_dim = min(size)
    aspect_ratio_max = max_shorter_dim / max_longer_dim

    if aspect_ratio <= aspect_ratio_max:
        # If the aspect ratio is less than max, we can resize based on the longer dim
        new_longer_dim = max_longer_dim
        new_shorter_dim = int(new_longer_dim * aspect_ratio)
    else:
        # If the aspect ratio is larger than max, we have to resize based on the shorter dim
        new_shorter_dim = max_shorter_dim
        new_longer_dim = int(new_shorter_dim / aspect_ratio)

    landscape = image.width > image.height
    if landscape:
        new_width = new_longer_dim
        new_height = new_shorter_dim
    else:
        new_width = new_shorter_dim
        new_height = new_longer_dim

    # Don't resize image if it's already within requirements
    if image.width <= new_width and image.height <= new_height:
        return image

    resized_image = image.resize((new_width, new_height))
    return resized_image


# TODO Probably worth putting into PDFToImageContent as a private method
#      Hmm might need to reuse in the DocumentToImageContent converter
def read_image_from_pdf(
    bytestream: ByteStream,
    page_range: List[int],
    size: Union[Tuple[int, int], Dict[float, Tuple[int, int]]],
    downsize: bool = False,
) -> List[str]:
    """
    Convert PDF file into a list of base64 encoded images.

    Checks PDF dimensions and adjusts size constraints based on aspect ratio.

    :param bytestream: ByteStream object containing the PDF data
    :param page_range: List of page numbers to convert to images
    :param size: Target size of the image. Tuple of (num_pixels, num_pixels)
        or a dictionary with aspect ratios as keys and tuples of (num_pixels, num_pixels) as values
    :param downsize: Whether to downsize the image
    :return:
        - List of Base64 strings
    """
    pypdf_and_pdf2image_import.check()

    pdf = PdfReader(BytesIO(bytestream.data))
    if not pdf.pages:
        logger.error("PDF file is empty: {file_path}", file_path=bytestream.meta.get("file_path"))
        return []

    all_pdf_images = []
    dpi = 300
    for page_number in page_range:
        # Get dimensions of the page
        page = pdf.pages[max(page_number - 1, 0)]  # Adjust for 0-based indexing
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        aspect_ratio = width / height

        # Calculate potential pixels for 300 dpi
        potential_pixels = (width * dpi / 72) * (height * 300 / 72)

        # TODO Can we dynamically load PIL's default limit?
        # 90% of PIL's default limit to prevent borderline cases
        pixel_limit = 89478485 * 0.9  # PIL's default limit * 0.9 margin factor

        conversion_args: Dict[str, Any] = {
            "dpi": dpi,
            "first_page": page_number,
            "last_page": page_number,
        }

        if potential_pixels > pixel_limit:
            logger.info(
                "Large PDF detected ({pixels:.2f} pixels, aspect ratio: {ratio:.2f}). "
                "Resizing the image to fit the pixel limit.",
                pixels=potential_pixels,
                ratio=aspect_ratio,
            )

            # For wide images (aspect ratio > 1), resize based on height while maintaining aspect ratio
            if aspect_ratio > 1:
                max_height = int((pixel_limit / aspect_ratio) ** 0.5)
                conversion_args["size"] = (None, max_height)
            # For tall images (aspect ratio < 1), resize based on width while maintaining aspect ratio
            else:
                max_width = int((pixel_limit * aspect_ratio) ** 0.5)
                conversion_args["size"] = (max_width, None)

        pdf_images: List[Image] = pdf2image.convert_from_bytes(bytestream.data, **conversion_args)

        # TODO Why do we downsize again, might as well do it in the conversion step
        image = pdf_images[0]
        if downsize:
            image = downsize_image(image, size)
        base64_image = to_base64_str(image)

        all_pdf_images.append(base64_image)

    return all_pdf_images
