# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_experimental.components.image_converters.document_to_image import DocumentToImageContent


class TestDocumentToImageContent:
    def test_to_dict(self) -> None:
        converter = DocumentToImageContent()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"file_path_meta_field": "file_path", "root_path": "", "detail": None, "size": None},
            "type": "haystack_experimental.components.image_converters.document_to_image.DocumentToImageContent",
        }

    def test_to_dict_not_defaults(self) -> None:
        converter = DocumentToImageContent(file_path_meta_field="image_path", root_path="/data", detail="high", size=(800, 600))
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {
                "file_path_meta_field": "image_path", "root_path": "/data", "detail": "high", "size": (800, 600)
            },
            "type": "haystack_experimental.components.image_converters.document_to_image.DocumentToImageContent",
        }

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {
                "file_path_meta_field": "image_path", "root_path": "/test", "detail": "auto", "size": (512, 512)
            },
            "type": "haystack_experimental.components.image_converters.document_to_image.DocumentToImageContent",
        }
        converter = component_from_dict(DocumentToImageContent, data, "name")
        assert component_to_dict(converter, "converter") == data

    def test_run_with_empty_documents_list(self) -> None:
        converter = DocumentToImageContent()
        results = converter.run(documents=[])
        assert results == {"image_documents": [], "image_contents": []}

    def test_run_with_missing_file_path_metadata(self) -> None:
        converter = DocumentToImageContent()
        # Document without file_path in metadata
        doc_no_path = Document(content="test", meta={})
        # Document with file_path but file doesn't exist
        doc_no_file = Document(content="test", meta={"file_path": "nonexistent.jpg"})
        with pytest.raises(ValueError, match="is missing the 'file_path' key"):
            _ = converter.run(documents=[doc_no_path, doc_no_file])

    def test_run_with_non_image_documents(self) -> None:
        converter = DocumentToImageContent()
        docx_doc = Document(content="test", meta={"file_path": "test/test_files/docx/sample_docx.docx"})
        with pytest.raises(ValueError, match="has an unsupported MIME type"):
            _ = converter.run(documents=[docx_doc])

    def test_run_with_invalid_file_path(self, caplog) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "wrong_name.jpg"})
        with pytest.raises(ValueError, match="has an invalid file path 'wrong_name.jpg'"):
            _ = converter.run(documents=[pdf_doc])

    def test_run_with_pdf_missing_page_number(self, caplog) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf"})
        with pytest.raises(ValueError, match="is missing the 'page_number' key"):
            _ = converter.run(documents=[pdf_doc])

    def test_run_with_image_documents(self) -> None:
        converter = DocumentToImageContent(root_path="test/test_files/images")
        image_doc = Document(content="test", meta={"file_path": "apple.jpg"})
        results = converter.run(documents=[image_doc])
        assert len(results["image_documents"]) == 1
        assert len(results["image_contents"]) == 1
        assert image_doc == results["image_documents"][0]

    def test_run_with_pdf_documents(self) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf", "page_number": 1})
        results = converter.run(documents=[pdf_doc])
        assert len(results["image_documents"]) == 1
        assert len(results["image_contents"]) == 1
        assert pdf_doc == results["image_documents"][0]

    def test_run_with_mixed_document_types(self) -> None:
        converter = DocumentToImageContent(root_path="test/test_files")
        documents = [
            Document(content="", meta={"file_path": "images/apple.jpg"}),
            Document(content="", meta={"file_path": "pdf/sample_pdf_1.pdf", "page_number": 1}),
            Document(content="text", meta={"file_path": "docx/sample_docx.docx"}),
        ]
        with pytest.raises(ValueError, match="has an unsupported MIME type"):
            _ = converter.run(documents=documents)
