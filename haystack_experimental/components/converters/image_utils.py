# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack import logging
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install pypdf pdf2image'") as pypdf_and_pdf2image_import:
    import pdf2image
    from pypdf import PdfReader

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image as PILImage
    from PIL.Image import Image
    from PIL.ImageFile import ImageFile


logger = logging.getLogger(__name__)


DETAIL_TO_IMAGE_SIZE = {"low": (512, 512), "high": (768, 2_048), "auto": (768, 2_048)}

# TODO We have to rely on this since our util functions are using the bytestream object.
#      We could change this to use the file path instead, where the file extension is used to determine the format.
# This is a mapping of image formats to their MIME types.
# from PIL import Image
# Image.init()  # <- Must force all plugins to initialize to get this mapping
# print(Image.MIME)
FORMAT_TO_MIME = {
    "BMP": "image/bmp",
    "DIB": "image/bmp",
    "PCX": "image/x-pcx",
    "EPS": "application/postscript",
    "GIF": "image/gif",
    "PNG": "image/png",
    "JPEG2000": "image/jp2",
    "ICNS": "image/icns",
    "ICO": "image/x-icon",
    "JPEG": "image/jpeg",
    "MPEG": "video/mpeg",
    "TIFF": "image/tiff",
    "MPO": "image/mpo",
    "PALM": "image/palm",
    "PDF": "application/pdf",
    "PPM": "image/x-portable-anymap",
    "PSD": "image/vnd.adobe.photoshop",
    "SGI": "image/sgi",
    "TGA": "image/x-tga",
    "WEBP": "image/webp",
    "XBM": "image/xbm",
    "XPM": "image/xpm",
}
MIME_TO_FORMAT = {v: k for k, v in FORMAT_TO_MIME.items()}
# Adding some common MIME types that are not in the PIL mapping
MIME_TO_FORMAT["image/jpg"] = "JPEG"


def open_image_to_base64(
    bytestream: ByteStream,
    size: Optional[Tuple[int, int]] = None,
    downsize: bool = False,
) -> str:
    """
    Open an image from a file path.

    :param bytestream: ByteStream object containing the image data
    :param size: Tuple of (num_pixels, num_pixels)
    :param downsize: If True, the image will be downscaled to the specified size.
    :return: Base64 string of the image
    """
    if not downsize:
        return base64.b64encode(bytestream.data).decode("utf-8")

    # Check the import
    pillow_import.check()

    # Load the image
    if bytestream.mime_type and bytestream.mime_type in MIME_TO_FORMAT:
        formats = [MIME_TO_FORMAT[bytestream.mime_type]]
    else:
        formats = None
    image: "ImageFile" = PILImage.open(BytesIO(bytestream.data), formats=formats)

    # TODO Probably should defer to image.get_format_mimetype() and print warning if bytestream.mime_type is not None
    #      and doesn't match?
    resolved_mime_type = bytestream.mime_type or image.get_format_mimetype()

    # Downsize the image
    downsized_image: "Image" = downsize_image(image, size)

    # Convert the image to base64 string
    if not resolved_mime_type:
        logger.warning(
            "Could not determine mime type for image. Defaulting to 'image/jpeg'. "
            "Consider providing a mime_type parameter."
        )
        resolved_mime_type = "image/jpeg"
    return pil_image_to_base64_str(downsized_image, mime_type=resolved_mime_type)


def pil_image_to_base64_str(image: Union["Image", "ImageFile"], mime_type: str = "image/jpeg") -> str:
    """
    Convert PIL Image to base64 string based on the specified mime type.

    :param image: PIL Image object
    :param mime_type: The mime type to save the image in. Default is "image/jpeg".
    :return: Base64 string of the image
    """
    # Convert image to RGB if it has an alpha channel and we are saving as JPEG
    if (mime_type == "image/jpeg" or mime_type == "image/jpg") and (
        image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    ):
        image = image.convert("RGB")

    buffered = BytesIO()
    form = MIME_TO_FORMAT.get(mime_type)
    if form is None:
        logger.warning(
            "Could not determine format for mime type {mime_type}. Defaulting to JPEG.",
            mime_type=mime_type,
        )
        form = "JPEG"
    image.save(buffered, format=form)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def downsize_image(
    image: Union["Image", "ImageFile"],
    size: Tuple[int, int],
) -> "Image":
    """
    Resize the image to be smaller while maintaining the aspect ratio.

    The latency of vision models can be improved by downsizing images ahead of time to be less than the
    maximum size they are expected to be. You can specify the size as a tuple of (num_pixels, num_pixels) or
    as a dictionary with tuples of (num_pixels, num_pixels) as values and aspect ratios as keys. The value of the
    key being closest to the image aspect ratio is used.

    :param image: PIL Image object
    :param size: Tuple of (num_pixels, num_pixels)
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


def read_image_from_pdf(
    bytestream: ByteStream,
    page_range: Optional[List[int]],
    size: Optional[Union[Tuple[int, int]]] = None,
    downsize: bool = False,
) -> List[str]:
    """
    Convert PDF file into a list of base64 encoded images.

    Checks PDF dimensions and adjusts size constraints based on aspect ratio.

    :param bytestream: ByteStream object containing the PDF data
    :param page_range: List of page numbers to convert to images
    :param size: Target size of the image. Tuple of (num_pixels, num_pixels).
    :param downsize: Whether to downsize the image
    :return:
        - List of Base64 strings
    """
    pypdf_and_pdf2image_import.check()

    try:
        pdf = PdfReader(BytesIO(bytestream.data))
    except Exception as e:
        logger.warning(
            "Could not read PDF file {file_path}. Skipping it. Error: {error}",
            file_path=bytestream.meta.get("file_path"),
            error=e,
        )
        return []

    if not pdf.pages:
        logger.warning("PDF file is empty: {file_path}", file_path=bytestream.meta.get("file_path"))
        return []

    all_pdf_images = []
    dpi = 300
    resolved_page_range = page_range or range(1, len(pdf.pages) + 1)
    num_pages = len(pdf.pages)
    for page_number in resolved_page_range:
        if page_number < 1 or page_number > num_pages:
            logger.warning(
                "Page {page_number} is out of range for the PDF file. Skipping it.",
                page_number=page_number,
            )
            continue

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

        image = pdf_images[0]
        if downsize and size is not None:
            image = downsize_image(image, size)

        # We always convert PDF to JPG
        base64_image = pil_image_to_base64_str(image, mime_type="image/jpeg")

        all_pdf_images.append(base64_image)

    return all_pdf_images
