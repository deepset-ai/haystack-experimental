# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

from haystack import logging
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport

from haystack_experimental.dataclasses.image_content import MIME_TO_FORMAT

with LazyImport("Run 'pip install pypdfium2'") as pypdfium2_import:
    from pypdfium2 import PdfDocument

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image as PILImage
    from PIL.Image import Image
    from PIL.ImageFile import ImageFile


logger = logging.getLogger(__name__)


def encode_image_to_base64(
    bytestream: ByteStream,
    size: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[str], str]:
    """
    Encode an image from a ByteStream into a base64-encoded string.

    Optionally resize the image before encoding to improve performance for downstream processing.

    :param bytestream: ByteStream containing the image data.
    :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
        maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
        when working with models that have resolution constraints or when transmitting images to remote services.

    :returns:
        A tuple (mime_type, base64_str), where:
        - mime_type (Optional[str]): The mime type of the encoded image, determined from the original data or image
          content. Can be None if the mime type cannot be reliably identified.
        - base64_str (str): The base64-encoded string representation of the (optionally resized) image.
    """
    if size is None:
        if bytestream.mime_type is None:
            logger.warning(
                "No mime type provided for the image. "
                "This may cause compatibility issues with downstream systems requiring a specific mime type. "
                "Please provide a mime type for the image."
            )
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
    size: Optional[Tuple[int, int]] = None,
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
    pypdfium2_import.check()
    pillow_import.check()

    try:
        pdf = PdfDocument(BytesIO(bytestream.data))
    except Exception as e:
        logger.warning(
            "Could not read PDF file {file_path}. Skipping it. Error: {error}",
            file_path=bytestream.meta.get("file_path"),
            error=e,
        )
        return []

    num_pages = len(pdf)
    if num_pages == 0:
        logger.warning("PDF file is empty: {file_path}", file_path=bytestream.meta.get("file_path"))
        pdf.close()
        return []

    all_pdf_images = []

    resolved_page_range = page_range or range(1, num_pages + 1)

    for page_number in resolved_page_range:
        if page_number < 1 or page_number > num_pages:
            logger.warning(
                "Page {page_number} is out of range for the PDF file. Skipping it.",
                page_number=page_number,
            )
            continue

        # Get dimensions of the page
        page = pdf[max(page_number - 1, 0)]  # Adjust for 0-based indexing
        _, _, width, height = page.get_mediabox()

        target_resolution_dpi = 300.0

        # From pypdfium2 docs: scale (float) – A factor scaling the number of pixels per PDF canvas unit. This defines
        # the resolution of the image. To convert a DPI value to a scale factor, multiply it by the size of 1 canvas
        # unit in inches (usually 1/72in).
        # https://pypdfium2.readthedocs.io/en/stable/python_api.html#pypdfium2._helpers.page.PdfPage.render
        target_scale = target_resolution_dpi / 72.0

        # Calculate potential pixels for target_dpi
        pixels_for_target_scale = width * height * target_scale**2

        pil_max_pixels = PILImage.MAX_IMAGE_PIXELS or int(1024 * 1024 * 1024 // 4 // 3)
        # 90% of PIL's default limit to prevent borderline cases
        pixel_limit = pil_max_pixels * 0.9

        scale = target_scale
        if pixels_for_target_scale > pixel_limit:
            logger.info(
                "Large PDF detected ({pixels:.2f} pixels). Resizing the image to fit the pixel limit.",
                pixels=pixels_for_target_scale,
            )
            scale = (pixel_limit / (width * height)) ** 0.5

        pdf_bitmap = page.render(scale=scale)

        image: "Image" = pdf_bitmap.to_pil()
        pdf_bitmap.close()
        if size is not None:
            image = resize_image_preserving_aspect_ratio(image, size)

        # We always convert PDF to JPG
        base64_image = encode_pil_image_to_base64(image, mime_type="image/jpeg")

        all_pdf_images.append((page_number, base64_image))

    pdf.close()

    return all_pdf_images
