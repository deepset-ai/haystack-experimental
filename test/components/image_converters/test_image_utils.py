# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from haystack.components.converters.utils import get_bytestream_from_source
from haystack.dataclasses import ByteStream
from PIL import Image
from pytest import LogCaptureFixture

from haystack_experimental.components.image_converters.image_utils import (
    convert_pdf_to_images,
    encode_image_to_base64,
    encode_pil_image_to_base64,
    resize_image_preserving_aspect_ratio,
)


class TestToBase64Jpeg:
    def test_to_base64_jpeg(self) -> None:
        image_array = np.array(
            [
                [[34.215402, 132.78745697, 71.04739979], [24.23156181, 35.26147199, 124.95610316]],
                [[155.47443501, 196.98050276, 154.74734292], [253.24590033, 84.62392497, 157.34396641]],
            ]
        )
        image = Image.fromarray(image_array.astype("uint8"))
        b64_str = encode_pil_image_to_base64(image=image, mime_type="image/jpeg")
        assert (
            b64_str
            == "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCe50jTTdTE6daffb/livr9KKKK9aHwo9ql8EfRH//Z"
        )


class TestDownsizeImage:
    def test_downsize_image_low_square(self) -> None:
        image_array = np.random.rand(768, 768, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        downsized_image = resize_image_preserving_aspect_ratio(image=image, size=(512, 512))
        assert downsized_image.width == 512
        assert downsized_image.height == 512

    def test_downsize_image_low_portrait(self) -> None:
        image_array = np.random.rand(2048, 1024, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        downsized_image = resize_image_preserving_aspect_ratio(image=image, size=(512, 512))
        assert downsized_image.width == 256
        assert downsized_image.height == 512

    def test_downsize_image_low_landscape(self) -> None:
        image_array = np.random.rand(1024, 2048, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        downsized_image = resize_image_preserving_aspect_ratio(image=image, size=(512, 512))
        assert downsized_image.width == 512
        assert downsized_image.height == 256

    def test_downsize_image_high_square(self) -> None:
        image_array = np.random.rand(2048, 2048, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        downsized_image = resize_image_preserving_aspect_ratio(image=image, size=(768, 2_048))
        assert downsized_image.width == 768
        assert downsized_image.height == 768

    def test_downsize_image_high_portrait(self) -> None:
        image_array = np.random.rand(2048, 1024, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        downsized_image = resize_image_preserving_aspect_ratio(image=image, size=(768, 2_048))
        assert downsized_image.width == 768
        assert downsized_image.height == 1536

    def test_downsize_image_high_landscape(self) -> None:
        image_array = np.random.rand(1024, 2048, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        downsized_image = resize_image_preserving_aspect_ratio(image=image, size=(768, 2_048))
        assert downsized_image.width == 1536
        assert downsized_image.height == 768

    def test_downsize_image_no_change(self) -> None:
        image_array = np.random.rand(256, 256, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        downsized_image = resize_image_preserving_aspect_ratio(image=image, size=(512, 512))
        assert downsized_image.width == 256
        assert downsized_image.height == 256


class TestReadImageFromPdf:
    def test_read_image_from_pdf(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))
        image = convert_pdf_to_images(bytestream=bytestream, page_range=[1])
        assert image is not None

    def test_read_image_from_pdf_with_size(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))
        image = convert_pdf_to_images(bytestream=bytestream, page_range=[1], size=(100, 100))
        assert image is not None

    def test_read_image_from_pdf_invalid_page(self, caplog: LogCaptureFixture) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))
        out = convert_pdf_to_images(bytestream=bytestream, page_range=[5])
        assert out == []
        assert "Page 5 is out of range for the PDF file. Skipping it." in caplog.text

    def test_read_image_from_pdf_error_reading_file(self, caplog: LogCaptureFixture) -> None:
        bytestream = ByteStream(
            data=b"", mime_type="application/pdf")
        out = convert_pdf_to_images(bytestream, [1])
        assert out == []
        assert "Could not read PDF file" in caplog.text

    def test_read_image_from_pdf_empty_file(self, caplog: LogCaptureFixture) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))

        with patch("haystack_experimental.components.image_converters.image_utils.PdfDocument") as mock_pdf_document:
            mock_pdf_document.__len__.return_value = 0
            out = convert_pdf_to_images(bytestream, [1])

        assert out == []
        assert "PDF file is empty" in caplog.text

    def test_scale_if_large_pdf(self, caplog: LogCaptureFixture) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))

        caplog.set_level(logging.INFO)

        mock_pdf_document = MagicMock()
        mock_pdf_document.__len__.return_value = 1
        mock_page = MagicMock()
        mock_page.get_mediabox.return_value = (0, 0, 1e6, 1e6)
        mock_pdf_document.__getitem__.return_value = mock_page

        with patch("haystack_experimental.components.image_converters.image_utils.PdfDocument", return_value=mock_pdf_document):
            convert_pdf_to_images(bytestream, [1])

        assert "Large PDF detected" in caplog.text


class TestOpenImageToBase64:
    def test_open_image_to_base64(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/images/haystack-logo.png"))
        base64_str = encode_image_to_base64(bytestream=bytestream)
        assert base64_str is not None

    def test_open_image_to_base64_downsize(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/images/haystack-logo.png"))
        base64_str = encode_image_to_base64(bytestream=bytestream, size=(128, 128))
        assert base64_str is not None
