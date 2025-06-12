# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import glob
import pytest
from haystack.components.converters.utils import get_bytestream_from_source
from haystack.dataclasses import ByteStream, Document
from PIL import Image
from pytest import LogCaptureFixture

from haystack_experimental.components.image_converters.image_utils import (
    _convert_pdf_to_base64_images,
    _encode_image_to_base64,
    _encode_pil_image_to_base64,
    _batch_convert_pdf_pages_to_images,
    _PdfPageInfo,
    _extract_image_sources_info,
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
        b64_str = _encode_pil_image_to_base64(image=image, mime_type="image/jpeg")
        assert (
            b64_str
            == "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAACAAIDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwCe50jTTdTE6daffb/livr9KKKK9aHwo9ql8EfRH//Z"
        )


class TestDownsizeImage:
    """
    In this test class, we are testing PIL Image.thumbnail() method to ensure that it conforms to our expectations.
    """
    def test_downsize_image_low_square(self) -> None:
        width, height = 768, 768
        image_array = np.random.rand(height, width, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        image.thumbnail(size=(512, 512), reducing_gap=None)
        assert image.width == 512
        assert image.height == 512

    def test_downsize_image_low_portrait(self) -> None:
        width, height = 1024, 2048
        image_array = np.random.rand(height, width, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        image.thumbnail(size=(512, 512), reducing_gap=None)
        assert image.width == 256
        assert image.height == 512

    def test_downsize_image_low_landscape(self) -> None:
        width, height = 2048, 1024
        image_array = np.random.rand(height, width, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        image.thumbnail(size=(512, 512), reducing_gap=None)
        assert image.width == 512
        assert image.height == 256

    def test_downsize_image_high_square(self) -> None:
        width, height = 2048, 2048
        image_array = np.random.rand(height, width, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        image.thumbnail(size=(768, 2_048), reducing_gap=None)
        assert image.width == 768
        assert image.height == 768

    def test_downsize_image_high_portrait(self) -> None:
        width, height = 1024, 2048
        image_array = np.random.rand(height, width, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        image.thumbnail(size=(768, 2_048), reducing_gap=None)
        assert image.width == 768
        assert image.height == 1536

    def test_downsize_image_high_landscape(self) -> None:
        width, height = 2048, 1024
        image_array = np.random.rand(height, width, 3) * 255
        image = Image.fromarray(image_array.astype("uint8"))
        image.thumbnail(size=(768, 2_048), reducing_gap=None)
        assert image.width == 768
        assert image.height == 384

    def test_downsize_image_no_change(self) -> None:
        width, height = 256, 256
        image_array = np.random.rand(height, width, 3) * 25
        image = Image.fromarray(image_array.astype("uint8"))
        image.thumbnail(size=(512, 512), reducing_gap=None)
        assert image.width == 256
        assert image.height == 256


class TestReadImageFromPdf:
    def test_read_image_from_pdf(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))
        image = _convert_pdf_to_base64_images(bytestream=bytestream, page_range=[1])
        assert image is not None

    def test_read_image_from_pdf_with_size(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))
        image = _convert_pdf_to_base64_images(bytestream=bytestream, page_range=[1], size=(100, 100))
        assert image is not None

    def test_read_image_from_pdf_invalid_page(self, caplog: LogCaptureFixture) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))
        out = _convert_pdf_to_base64_images(bytestream=bytestream, page_range=[5])
        assert out == []
        assert "Page 5 is out of range for the PDF file. Skipping it." in caplog.text

    def test_read_image_from_pdf_error_reading_file(self, caplog: LogCaptureFixture) -> None:
        bytestream = ByteStream(
            data=b"", mime_type="application/pdf")
        out = _convert_pdf_to_base64_images(bytestream, [1])
        assert out == []
        assert "Could not read PDF file" in caplog.text

    def test_read_image_from_pdf_empty_file(self, caplog: LogCaptureFixture) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/pdf/sample_pdf_1.pdf"))

        with patch("haystack_experimental.components.image_converters.image_utils.PdfDocument") as mock_pdf_document:
            mock_pdf_document.__len__.return_value = 0
            out = _convert_pdf_to_base64_images(bytestream, [1])

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
            _convert_pdf_to_base64_images(bytestream, [1])

        assert "Large PDF detected" in caplog.text


class TestOpenImageToBase64:
    def test_open_image_to_base64(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/images/haystack-logo.png"))
        base64_str = _encode_image_to_base64(bytestream=bytestream)
        assert base64_str is not None

    def test_open_image_to_base64_downsize(self) -> None:
        bytestream = get_bytestream_from_source(Path("test/test_files/images/haystack-logo.png"))
        base64_str = _encode_image_to_base64(bytestream=bytestream, size=(128, 128))
        assert base64_str is not None


class TestExtractImageSourcesInfo:
    def test_extract_image_source_info(self, test_files_path):

        image_paths = glob.glob(str(test_files_path / "images" / "*.*")) + glob.glob(str(test_files_path / "pdf" / "*.pdf"))

        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        images_source_info = _extract_image_sources_info(documents=documents, file_path_meta_field="file_path", root_path="")
        assert len(images_source_info) == len(documents)

        for image_source_info in images_source_info:
            assert str(image_source_info["path"]) in image_paths
            assert image_source_info["mime_type"] in ["image/jpeg", "image/png", "application/pdf"]
            if image_source_info["mime_type"] == "application/pdf":
                assert image_source_info.get("page_number") == 1
            else:
                assert "page_number" not in image_source_info

    def test_extract_image_source_info_errors(self, test_files_path):

        document = Document(content="test")
        with pytest.raises(ValueError, match="missing the 'file_path' key"):
            _extract_image_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")

        document = Document(content="test", meta={"file_path": "invalid_path"})
        with pytest.raises(ValueError, match="has an invalid file path"):
            _extract_image_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")

        document = Document(content="test", meta={"file_path": str(test_files_path / "docx" / "sample_docx.docx")})
        with pytest.raises(ValueError, match="has an unsupported MIME type"):
            _extract_image_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")

        document = Document(content="test", meta={"file_path": str(test_files_path / "pdf" / "sample_pdf_1.pdf")})
        with pytest.raises(ValueError, match="missing the 'page_number' key"):
            _extract_image_sources_info(documents=[document], file_path_meta_field="file_path", root_path="")

class TestBatchConvertPdfPagesToImages:
    @patch("haystack_experimental.components.image_converters.image_utils._convert_pdf_to_pil_images")
    def test_batch_convert_pdf_pages_to_images(self, mocked_convert_pdf_to_pil_images, test_files_path):

        mocked_convert_pdf_to_pil_images.return_value = [(1, Image.new("RGB", (100, 100))), (2, Image.new("RGB", (100, 100)))]

        pdf_path = test_files_path / "pdf" / "sample_pdf_1.pdf"
        pdf_doc_1: _PdfPageInfo = {"doc_idx": 0, "path": pdf_path, "page_number": 1}
        pdf_doc_2: _PdfPageInfo = {"doc_idx": 1, "path": pdf_path, "page_number": 2}
        pdf_documents = [pdf_doc_1, pdf_doc_2]

        result = _batch_convert_pdf_pages_to_images(pdf_page_infos=pdf_documents, return_base64=False)

        pdf_bytestream = ByteStream.from_file_path((pdf_path))

        mocked_convert_pdf_to_pil_images.assert_called_once_with(
            bytestream=pdf_bytestream,
            page_range=[1, 2],
            size=None
        )

        assert len(result) == len(pdf_documents)
        assert 0 in result and 1 in result
        assert isinstance(result[0], Image.Image)
        assert isinstance(result[1], Image.Image)

    @patch("haystack_experimental.components.image_converters.image_utils._convert_pdf_to_base64_images")
    def test_batch_convert_pdf_pages_to_images_base64(self, mocked_convert_pdf_to_base64_images, test_files_path):

        mocked_convert_pdf_to_base64_images.return_value = [(1, "base64_image_1"), (2, "base64_image_2")]

        pdf_path = test_files_path / "pdf" / "sample_pdf_1.pdf"
        pdf_doc_1: _PdfPageInfo = {"doc_idx": 0, "path": pdf_path, "page_number": 1}
        pdf_doc_2: _PdfPageInfo = {"doc_idx": 1, "path": pdf_path, "page_number": 2}
        pdf_documents = [pdf_doc_1, pdf_doc_2]

        result = _batch_convert_pdf_pages_to_images(pdf_page_infos=pdf_documents, return_base64=True)

        pdf_bytestream = ByteStream.from_file_path((pdf_path))

        mocked_convert_pdf_to_base64_images.assert_called_once_with(
            bytestream=pdf_bytestream,
            page_range=[1, 2],
            size=None
        )

        assert result == {0: "base64_image_1", 1: "base64_image_2"}

    def test_batch_convert_pdf_pages_to_images_no_pages_info(self):
        result = _batch_convert_pdf_pages_to_images(pdf_page_infos=[])

        assert isinstance(result, dict)
        assert len(result) == 0
