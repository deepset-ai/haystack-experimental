# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict

from haystack_experimental.components.image_converters.document_to_image import DocumentToImageContent


class TestDocumentToImageContent:
    def test_to_dict(self) -> None:
        converter = DocumentToImageContent()
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"root_path": None, "detail": None, "size": None},
            "type": "haystack_experimental.components.image_converters.document_to_image.DocumentToImageContent",
        }

    def test_to_dict_not_defaults(self) -> None:
        converter = DocumentToImageContent(root_path="/data", detail="high", size=(800, 600))
        assert component_to_dict(converter, "converter") == {
            "init_parameters": {"root_path": "/data", "detail": "high", "size": (800, 600)},
            "type": "haystack_experimental.components.image_converters.document_to_image.DocumentToImageContent",
        }

    def test_from_dict(self) -> None:
        data = {
            "init_parameters": {"root_path": "/test", "detail": "auto", "size": (512, 512)},
            "type": "haystack_experimental.components.image_converters.document_to_image.DocumentToImageContent",
        }
        converter = component_from_dict(DocumentToImageContent, data, "name")
        assert component_to_dict(converter, "converter") == data

    def test_run_with_empty_documents_list(self) -> None:
        converter = DocumentToImageContent()
        results = converter.run(documents=[])
        assert results == {
            "image_documents": [],
            "image_contents": [],
            "non_image_documents": []
        }

    def test_run_with_missing_file_path_metadata(self) -> None:
        converter = DocumentToImageContent()
        # Document without file_path in metadata
        doc_no_path = Document(content="test", meta={})
        # Document with file_path but file doesn't exist
        doc_no_file = Document(content="test", meta={"file_path": "nonexistent.jpg"})
        results = converter.run(documents=[doc_no_path, doc_no_file])
        assert len(results["image_documents"]) == 0
        assert len(results["image_contents"]) == 0
        assert len(results["non_image_documents"]) == 2
        assert doc_no_path in results["non_image_documents"]
        assert doc_no_file in results["non_image_documents"]

    def test_run_with_non_image_documents(self) -> None:
        converter = DocumentToImageContent()
        docx_doc = Document(content="test", meta={"file_path": "test/test_files/docx/sample_docx.docx"})
        results = converter.run(documents=[docx_doc])
        assert len(results["image_documents"]) == 0
        assert len(results["image_contents"]) == 0
        assert len(results["non_image_documents"]) == 1
        assert docx_doc == results["non_image_documents"][0]

    def test_run_with_pdf_missing_page_number(self, caplog) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "document.pdf"})
        results = converter.run(documents=[pdf_doc])
        assert len(results["image_documents"]) == 0
        assert len(results["image_contents"]) == 0
        assert len(results["non_image_documents"]) == 1
        assert pdf_doc in results["non_image_documents"]
        assert "missing a `file_path`" in caplog.text

    def test_run_with_image_documents(self) -> None:
        converter = DocumentToImageContent(root_path="test/test_files/images")
        image_doc = Document(content="test", meta={"file_path": "apple.jpg"})
        results = converter.run(documents=[image_doc])
        assert len(results["image_documents"]) == 1
        assert len(results["image_contents"]) == 1
        assert len(results["non_image_documents"]) == 0
        assert image_doc == results["image_documents"][0]

    def test_run_with_pdf_documents(self) -> None:
        converter = DocumentToImageContent()
        pdf_doc = Document(content="test", meta={"file_path": "test/test_files/pdf/sample_pdf_1.pdf", "page_number": 1})
        results = converter.run(documents=[pdf_doc])
        assert len(results["image_documents"]) == 1
        assert len(results["image_contents"]) == 1
        assert len(results["non_image_documents"]) == 0
        assert pdf_doc == results["image_documents"][0]

    def test_run_with_mixed_document_types(self) -> None:
        converter = DocumentToImageContent(root_path="test/test_files")
        documents = [
            Document(content="", meta={"file_path": "images/apple.jpg"}),
            Document(content="", meta={"file_path": "pdf/sample_pdf_1.pdf", "page_number": 1}),
            Document(content="text", meta={"file_path": "docx/sample_docx.docx"}),
        ]
        results = converter.run(documents=documents)
        assert len(results["image_documents"]) == 2
        assert len(results["image_contents"]) == 2
        assert len(results["non_image_documents"]) == 1

    def test_deduplicate_pdf_documents(self) -> None:
        doc1 = Document(content="First chunk text", meta={"file_path": "test.pdf", "page_number": 1})
        doc2 = Document(content="Second chunk text", meta={"file_path": "test.pdf", "page_number": 1})  # duplicate
        doc3 = Document(content="Page 2 text", meta={"file_path": "test.pdf", "page_number": 2})
        doc4 = Document(content="Other Page 1 text", meta={"file_path": "other.pdf", "page_number": 1})
        documents = [doc1, doc2, doc3, doc4]
        deduplicated = DocumentToImageContent._deduplicate(documents)
        assert len(deduplicated) == 3
        assert doc1 == deduplicated[0]
        assert doc2 not in deduplicated  # Should be removed as duplicate
        assert doc3 == deduplicated[1]
        assert doc4 == deduplicated[2]

    def test_run_logging_warning_for_missing_files(self, caplog) -> None:
        converter = DocumentToImageContent()
        doc_no_path = Document(content="", meta={})
        converter.run(documents=[doc_no_path])
        assert "missing a `file_path`" in caplog.text
        assert "1 Documents" in caplog.text
