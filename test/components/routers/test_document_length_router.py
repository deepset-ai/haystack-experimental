# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch
from pathlib import Path

from haystack_experimental.components.routers import DocumentLengthRouter
from haystack.dataclasses import Document
from haystack import Pipeline
from haystack.core.pipeline.base import component_to_dict, component_from_dict


class TestDocumentLengthRouter:
    def test_init(self):
        router = DocumentLengthRouter(
            threshold=20
        )
        assert router.threshold == 20

    def test_run(self):
        docs = [
            Document(content="Short"),
            Document(content="Long document "*20),
        ]
        router = DocumentLengthRouter(threshold=10)
        result = router.run(documents=docs)

        assert len(result["short_documents"]) == 1
        assert len(result["long_documents"]) == 1
        assert result["short_documents"][0] == docs[0]
        assert result["long_documents"][0] == docs[1]

    def test_run_with_null_content(self):
        docs = [
            Document(content=None),
            Document(content="Long document "*20),
        ]
        router = DocumentLengthRouter(threshold=10)
        result = router.run(documents=docs)

        assert len(result["short_documents"]) == 1
        assert len(result["long_documents"]) == 1
        assert result["short_documents"][0] == docs[0]
        assert result["long_documents"][0] == docs[1]

    def test_to_dict(self):
        router = DocumentLengthRouter(
            threshold=10
        )
        expected_dict = {
            "type": "haystack_experimental.components.routers.document_length_router.DocumentLengthRouter",
            "init_parameters": {
                "threshold": 10
            },
        }
        assert component_to_dict(router, "router") == expected_dict

    def test_from_dict(self):
        router_dict = {
            "type": "haystack_experimental.components.routers.document_length_router.DocumentLengthRouter",
            "init_parameters": {
                "threshold": 10
            },
        }
        loaded_router = component_from_dict(DocumentLengthRouter, router_dict, name="router")


        assert isinstance(loaded_router, DocumentLengthRouter)
        assert loaded_router.threshold == 10

    # def test_run_with_mime_type_meta_field(self):
    #     docs = [
    #         Document(content="Example text", meta={"mime_type": "text/plain"}),
    #         Document(content="Another document", meta={"mime_type": "application/pdf"}),
    #         Document(content="Audio content", meta={"mime_type": "audio/x-wav"}),
    #         Document(content="Unknown type", meta={"mime_type": "application/unknown"}),
    #     ]

    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=["text/plain", "application/pdf", "audio/x-wav"]
    #     )
    #     result = router.run(documents=docs)

    #     assert len(result["text/plain"]) == 1
    #     assert len(result["application/pdf"]) == 1
    #     assert len(result["audio/x-wav"]) == 1
    #     assert len(result["unclassified"]) == 1
    #     assert result["text/plain"][0].content == "Example text"
    #     assert result["application/pdf"][0].content == "Another document"
    #     assert result["audio/x-wav"][0].content == "Audio content"
    #     assert result["unclassified"][0].content == "Unknown type"

    # def test_run_with_file_path_meta_field(self):
    #     docs = [
    #         Document(content="Example text", meta={"file_path": "example.txt"}),
    #         Document(content="PDF document", meta={"file_path": "document.pdf"}),
    #         Document(content="Markdown content", meta={"file_path": "readme.md"}),
    #         Document(content="Unknown extension", meta={"file_path": "file.xyz"}),
    #     ]

    #     router = DocumentTypeRouter(
    #         file_path_meta_field="file_path",
    #         mime_types=["text/plain", "application/pdf", "text/markdown"]
    #     )
    #     result = router.run(documents=docs)

    #     assert len(result["text/plain"]) == 1
    #     assert len(result["application/pdf"]) == 1
    #     assert len(result["text/markdown"]) == 1
    #     assert len(result["unclassified"]) == 1

    # def test_run_with_both_meta_fields(self):
    #     docs = [
    #         Document(
    #             content="Text with explicit mime type", meta={"mime_type": "application/pdf", "file_path": "file.txt"}
    #         ),
    #         Document(content="Text inferred from path", meta={"file_path": "file.txt"}),
    #     ]

    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         file_path_meta_field="file_path",
    #         mime_types=["text/plain", "application/pdf"]
    #     )
    #     result = router.run(documents=docs)

    #     # First doc should be classified as PDF (explicit mime type)
    #     assert len(result["application/pdf"]) == 1
    #     assert result["application/pdf"][0].content == "Text with explicit mime type"

    #     # Second doc should be classified as text/plain (inferred from .txt)
    #     assert len(result["text/plain"]) == 1
    #     assert result["text/plain"][0].content == "Text inferred from path"

    # def test_run_with_missing_metadata(self):
    #     docs = [
    #         Document(content="No metadata"),
    #         Document(content="Empty meta", meta={}),
    #         Document(content="Wrong meta field", meta={"other_field": "value"}),
    #     ]

    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=["text/plain"]
    #     )
    #     result = router.run(documents=docs)

    #     assert len(result["unclassified"]) == 3
    #     assert "text/plain" not in result

    # def test_run_with_regex_patterns(self):
    #     docs = [
    #         Document(content="Plain text", meta={"mime_type": "text/plain"}),
    #         Document(content="HTML text", meta={"mime_type": "text/html"}),
    #         Document(content="Markdown text", meta={"mime_type": "text/markdown"}),
    #         Document(content="JPEG image", meta={"mime_type": "image/jpeg"}),
    #         Document(content="PNG image", meta={"mime_type": "image/png"}),
    #         Document(content="PDF document", meta={"mime_type": "application/pdf"}),
    #     ]

    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=[r"text/.*", r"image/.*"]
    #     )
    #     result = router.run(documents=docs)

    #     assert len(result[r"text/.*"]) == 3
    #     assert len(result[r"image/.*"]) == 2
    #     assert len(result["unclassified"]) == 1

    # def test_run_with_exact_matching(self):
    #     docs = [
    #         Document(content="Plain text", meta={"mime_type": "text/plain"}),
    #         Document(content="Markdown text", meta={"mime_type": "text/markdown"}),
    #         Document(content="PDF document", meta={"mime_type": "application/pdf"}),
    #     ]

    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=["text/plain", "application/pdf"]
    #     )
    #     result = router.run(documents=docs)

    #     assert len(result["text/plain"]) == 1
    #     assert len(result["application/pdf"]) == 1
    #     assert len(result["unclassified"]) == 1
    #     assert result["unclassified"][0].content == "Markdown text"

    # def test_run_with_empty_documents_list(self):
    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=["text/plain", "application/pdf"]
    #     )
    #     result = router.run(documents=[])

    #     assert len(result) == 0

    # def test_run_with_custom_mime_types(self):
    #     docs = [
    #         Document(content="Word document", meta={"file_path": "document.docx"}),
    #         Document(content="Markdown file", meta={"file_path": "readme.md"}),
    #         Document(content="Outlook message", meta={"file_path": "email.msg"}),
    #     ]

    #     router = DocumentTypeRouter(
    #         file_path_meta_field="file_path",
    #         mime_types=["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/markdown",
    #                     "application/vnd.ms-outlook"],
    #         additional_mimetypes={"application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx"}
    #     )
    #     result = router.run(documents=docs)

    #     assert len(result["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]) == 1
    #     assert len(result["text/markdown"]) == 1
    #     assert len(result["application/vnd.ms-outlook"]) == 1

    # @patch('mimetypes.guess_type')
    # def test_get_mime_type_with_mocked_mimetypes(self, mock_guess_type):
    #     mock_guess_type.return_value = ("text/plain", None)

    #     router = DocumentTypeRouter(
    #         file_path_meta_field="file_path",
    #         mime_types=["text/plain"]
    #     )

    #     mime_type = router._get_mime_type(Path("test.txt"))
    #     assert mime_type == "text/plain"
    #     mock_guess_type.assert_called_once()

    # def test_get_mime_type_with_custom_extensions(self):
    #     router = DocumentTypeRouter(
    #         file_path_meta_field="file_path",
    #         mime_types=["text/markdown"]
    #     )

    #     # Test markdown extension (should be in CUSTOM_MIMETYPES)
    #     mime_type = router._get_mime_type(Path("readme.md"))
    #     assert mime_type == "text/markdown"

    #     # Test .msg extension
    #     mime_type = router._get_mime_type(Path("email.msg"))
    #     assert mime_type == "application/vnd.ms-outlook"

    # def test_get_mime_type_case_insensitive(self):
    #     router = DocumentTypeRouter(
    #         file_path_meta_field="file_path",
    #         mime_types=["text/markdown"]
    #     )

    #     # Test uppercase extension
    #     mime_type = router._get_mime_type(Path("README.MD"))
    #     assert mime_type == "text/markdown"

    # def test_serde_in_pipeline(self):
    #     document_type_router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=["text/plain", "application/pdf"]
    #     )

    #     pipeline = Pipeline()
    #     pipeline.add_component(instance=document_type_router, name="document_type_router")

    #     pipeline_dict = pipeline.to_dict()

    #     expected_components = {
    #         "document_type_router": {
    #             "type": "haystack_experimental.components.routers.document_type_router.DocumentTypeRouter",
    #             "init_parameters": {
    #                 "mime_type_meta_field": "mime_type",
    #                 "file_path_meta_field": None,
    #                 "mime_types": ["text/plain", "application/pdf"],
    #                 "additional_mimetypes": None
    #             },
    #         }
    #     }

    #     assert pipeline_dict["components"] == expected_components

    #     pipeline_yaml = pipeline.dumps()
    #     new_pipeline = Pipeline.loads(pipeline_yaml)
    #     assert new_pipeline == pipeline

    # def test_run_preserves_document_metadata(self):
    #     docs = [
    #         Document(
    #             content="Test content",
    #             meta={
    #                 "mime_type": "text/plain",
    #                 "author": "John Doe",
    #                 "created_at": "2023-01-01",
    #                 "custom_field": "custom_value"
    #             }
    #         )
    #     ]

    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=["text/plain"]
    #     )
    #     result = router.run(documents=docs)

    #     classified_doc = result["text/plain"][0]
    #     assert classified_doc.meta["author"] == "John Doe"
    #     assert classified_doc.meta["created_at"] == "2023-01-01"
    #     assert classified_doc.meta["custom_field"] == "custom_value"
    #     assert classified_doc.meta["mime_type"] == "text/plain"

    # def test_run_with_multiple_matching_patterns(self):
    #     docs = [
    #         Document(content="Plain text file", meta={"mime_type": "text/plain"})
    #     ]

    #     # Both patterns should match text/plain, but only the first should be used
    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         mime_types=[r"text/.*", r"text/plain"]
    #     )
    #     result = router.run(documents=docs)

    #     assert len(result[r"text/.*"]) == 1
    #     assert r"text/plain" not in result or len(result[r"text/plain"]) == 0

    # def test_run_integration_example(self):
    #     docs = [
    #         Document(content="Example text", meta={"file_path": "example.txt"}),
    #         Document(content="Another document", meta={"mime_type": "application/pdf"}),
    #         Document(content="Unknown type")
    #     ]

    #     router = DocumentTypeRouter(
    #         mime_type_meta_field="mime_type",
    #         file_path_meta_field="file_path",
    #         mime_types=["text/plain", "application/pdf"]
    #     )

    #     result = router.run(documents=docs)

    #     assert len(result["text/plain"]) == 1
    #     assert len(result["application/pdf"]) == 1
    #     assert len(result["unclassified"]) == 1
