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


_OPENAI_DETAIL_TO_IMAGE_SIZE = {"low": (512, 512), "high": (768, 2_048), "auto": (768, 2_048)}

# NOTE: We have to rely on this since our util functions are using the bytestream object.
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


def encode_image_to_base64(
    bytestream: ByteStream,
    size: Optional[Tuple[int, int]] = None,
) -> Tuple[str, str]:
    """
    Encode an image from a ByteStream into a base64-encoded string.

    Optionally resize the image before encoding to improve performance for downstream processing.

    :param bytestream: ByteStream containing the image data.
    :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
        maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
        when working with models that have resolution constraints or when transmitting images to remote services.

    :returns:
        A tuple (mime_type, base64_str), where:
        - mime_type (str): The MIME type of the encoded image, determined from the original data or image content.
          Defaults to 'image/jpeg' if the type cannot be reliably identified.
        - base64_str (str): The base64-encoded string representation of the (optionally resized) image.
    """
    if size is None:
        return bytestream.mime_type, base64.b64encode(bytestream.data).decode("utf-8")

    # Check the import
    pillow_import.check()

    # Load the image
    if bytestream.mime_type and bytestream.mime_type in MIME_TO_FORMAT:
        formats = [MIME_TO_FORMAT[bytestream.mime_type]]
    else:
        formats = None
    image: "ImageFile" = PILImage.open(BytesIO(bytestream.data), formats=formats)

    # NOTE: We prefer the format returned by PIL
    inferred_mime_type = image.get_format_mimetype() or bytestream.mime_type

    # Downsize the image
    downsized_image: "Image" = resize_image_preserving_aspect_ratio(image, size)

    # Convert the image to base64 string
    if not inferred_mime_type:
        logger.warning(
            "Could not determine mime type for image. Defaulting to 'image/jpeg'. "
            "Consider providing a mime_type parameter."
        )
        inferred_mime_type = "image/jpeg"
    return inferred_mime_type, encode_pil_image_to_base64(downsized_image, mime_type=inferred_mime_type)


def encode_pil_image_to_base64(image: Union["Image", "ImageFile"], mime_type: str = "image/jpeg") -> str:
    """
    Convert a PIL Image object to a base64-encoded string.

    Automatically converts images with transparency to RGB if saving as JPEG.

    :param image: A PIL Image or ImageFile object to encode.
    :param mime_type: The MIME type to use when encoding the image. Defaults to "image/jpeg".
    :returns:
        Base64-encoded string representing the image.
    """
    # Check the import
    pillow_import.check()

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


def resize_image_preserving_aspect_ratio(
    image: Union["Image", "ImageFile"],
    size: Tuple[int, int],
) -> "Image":
    """
    Resizes the image to fit within the specified dimensions while maintaining the original aspect ratio.

    The image is only resized if its dimensions exceed the given maximum size.

    Downsizing images can be beneficial for both reducing latency and conserving memory. Some models have resolution
    limitations, so it's advisable to resize images to the maximum supported resolution before passing them to the
    model. This can improve latency when sending images to remote services and optimize memory usage.

    :param image: A PIL Image or ImageFile object to resize.
    :param size: Maximum allowed dimensions (width, height).
    :returns:
        A resized PIL Image object.
    """
    # Check the import
    pillow_import.check()

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


def convert_pdf_to_images(
    bytestream: ByteStream,
    page_range: Optional[List[int]] = None,
    size: Optional[Union[Tuple[int, int]]] = None,
) -> List[Tuple[int, str]]:
    """
    Convert PDF file into a list of base64 encoded images with the mime type "image/jpeg".

    Checks PDF dimensions and adjusts size constraints based on aspect ratio.

    :param bytestream: ByteStream object containing the PDF data
    :param page_range: List of page numbers and/or page ranges to convert to images. Page numbers start at 1.
        If None, all pages in the PDF will be converted. Pages outside the valid range (1 to number of pages)
        will be skipped with a warning. For example, page_range=[1, 3] will convert only the first and third
        pages of the document. It also accepts printable range strings, e.g.:  ['1-3', '5', '8', '10-12']
        will convert pages 1, 2, 3, 5, 8, 10, 11, 12.
    :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
        maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
        when working with models that have resolution constraints or when transmitting images to remote services.
    :returns:
        A list of tuples, each tuple containing the page number and the base64-encoded image string.
    """
    # Check the imports
    pypdf_and_pdf2image_import.check()
    pillow_import.check()

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

        pil_max_pixels = PILImage.MAX_IMAGE_PIXELS or int(1024 * 1024 * 1024 // 4 // 3)
        # 90% of PIL's default limit to prevent borderline cases
        pixel_limit = pil_max_pixels * 0.9

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
        if size is not None:
            image = resize_image_preserving_aspect_ratio(image, size)

        # We always convert PDF to JPG
        base64_image = encode_pil_image_to_base64(image, mime_type="image/jpeg")

        all_pdf_images.append((page_number, base64_image))

    return all_pdf_images
