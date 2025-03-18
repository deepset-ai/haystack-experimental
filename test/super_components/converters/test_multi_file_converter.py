import pytest

from haystack import Document
from haystack.dataclasses import ByteStream
from haystack_experimental.core.super_component import SuperComponent
from haystack_experimental.super_components.converters.multi_file_converter import MultiFileConverter


@pytest.fixture
def converter():
    return MultiFileConverter()


class TestMultiFileConverter:
    def test_init_default_params(self, converter):
        """Test initialization with default parameters"""
        assert converter.encoding == "utf-8"
        assert converter.json_content_key == "content"
        assert isinstance(converter, SuperComponent)

    def test_init_custom_params(self, converter):
        """Test initialization with custom parameters"""
        converter = MultiFileConverter(
            encoding="latin-1",
            json_content_key="text"
        )
        assert converter.encoding == "latin-1"
        assert converter.json_content_key == "text"

    def test_to_dict(self, converter):
        """Test serialization to dictionary"""
        data = converter.to_dict()
        assert data == {
            "type": "haystack_experimental.super_components.converters.multi_file_converter.MultiFileConverter",
            "init_parameters": {
                "encoding": "utf-8",
                "json_content_key": "content"
            }
        }

    def test_from_dict(self):
        """Test deserialization from dictionary"""
        data = {
            "type": "haystack_experimental.super_components.converters.multi_file_converter.MultiFileConverter",
            "init_parameters": {
                "encoding": "latin-1",
                "json_content_key": "text"
            }
        }
        conv = MultiFileConverter.from_dict(data)
        assert conv.encoding == "latin-1"
        assert conv.json_content_key == "text"

    @pytest.mark.parametrize(
        "suffix,file_path",
        [
            ("csv", "csv/sample_1.csv"),
            ("docx", "docx/sample_docx.docx"),
            ("html", "html/what_is_haystack.html"),
            ("json", "json/json_conversion_testfile.json"),
            ("md", "markdown/sample.md"),
            ("pdf", "pdf/sample_pdf_1.pdf"),
            ("pptx", "pptx/sample_pptx.pptx"),
            ("txt", "txt/doc_1.txt"),
            ("xlsx", "xlsx/table_empty_rows_and_columns.xlsx"),
        ]
    )
    @pytest.mark.integration
    def test_run(self, test_files_path, converter, suffix, file_path):
        unclassified_bytestream = ByteStream(b"unclassified content")
        unclassified_bytestream.meta["content_type"] = "unknown_type"

        paths = [test_files_path / file_path, unclassified_bytestream]

        output = converter.run(sources=paths)
        docs = output["documents"]
        unclassified = output["unclassified"]

        assert len(docs) == 1
        assert isinstance(docs[0], Document)
        assert docs[0].content is not None
        assert docs[0].meta["file_path"].endswith(suffix)

        assert len(unclassified) == 1
        assert isinstance(unclassified[0], ByteStream)
        assert unclassified[0].meta["content_type"] == "unknown_type"

    def test_run_with_meta(self, test_files_path, converter):
        """Test conversion with metadata"""
        paths = [test_files_path / "txt" / "doc_1.txt"]
        meta = {"language": "en", "author": "test"}
        output = converter.run(sources=paths, meta=meta)
        docs = output["documents"]
        assert docs[0].meta["language"] == "en"
        assert docs[0].meta["author"] == "test"

    def test_run_with_bytestream(self, test_files_path, converter):
        """Test converting ByteStream input"""
        bytestream = ByteStream(
            data=b"test content",
            mime_type="text/plain",
            meta={"file_path": "test.txt"}
        )
        output = converter.run(sources=[bytestream])
        docs = output["documents"]
        assert len(docs) == 1
        assert docs[0].content == "test content"
        assert docs[0].meta["file_path"] == "test.txt"

    def test_run_error_handling(self, test_files_path, converter, caplog):
        """Test error handling for non-existent files"""
        paths = [test_files_path / "non_existent.txt"]
        with caplog.at_level("WARNING"):
            output = converter.run(sources=paths)
            assert "Could not read" in caplog.text
            assert len(output["documents"]) == 0

    @pytest.mark.integration
    def test_run_all_file_types(self, test_files_path, converter):
        """Test converting all supported file types in parallel"""
        paths = [
            test_files_path / "csv" / "sample_1.csv",
            test_files_path / "docx" / "sample_docx.docx",
            test_files_path / "html" / "what_is_haystack.html",
            test_files_path / "json" / "json_conversion_testfile.json",
            test_files_path / "markdown" / "sample.md",
            test_files_path / "txt" / "doc_1.txt",
            test_files_path / "pdf" / "sample_pdf_1.pdf",
            test_files_path / "pptx" / "sample_pptx.pptx",
            test_files_path / "xlsx" / "table_empty_rows_and_columns.xlsx"
        ]
        output = converter.run(sources=paths)
        docs = output["documents"]

        # Verify we got a document for each file
        assert len(docs) == len(paths)
        assert all(isinstance(doc, Document) for doc in docs)
