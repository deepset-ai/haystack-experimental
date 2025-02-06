# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.converters import (
    CSVToDocument,
    DOCXToDocument,
    HTMLToDocument,
    JSONConverter,
    MarkdownToDocument,
    PPTXToDocument,
    PyPDFToDocument,
    TextFileToDocument,
    XLSXToDocument,
)
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors.document_splitter import DocumentSplitter, Language
from haystack.components.routers import FileTypeRouter
from haystack.utils import deserialize_callable, serialize_callable

from haystack_experimental.core.super_component import SuperComponentBase


class ConverterMimeType(str, Enum):
    CSV = "text/csv"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"
    JSON = "application/json"
    MD = "text/markdown"
    TEXT = "text/plain"
    PDF = "application/pdf"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@component
class AutoFileConverter(SuperComponentBase):
    """
    A file converter that handles multiple file types and their pre-processing.

    The AutoFileConverter handles the following file types:
    - CSV
    - DOCX
    - HTML
    - JSON
    - MD
    - TEXT
    - PDF (no OCR)
    - PPTX
    - XLSX

    It splits all non-tabular data into Documents as specified by the splitting parameters.
    Tabular data (CSV & XLSX) is returned without splitting.

    Usage:
    ```
    converter = AutoFileConverter()
    converter.run(sources=["test.txt", "test.pdf"], meta={})
    ```
    """

    def __init__( # noqa: PLR0915
        self,
        split_by: Literal["function", "page", "passage", "period", "word", "line", "sentence"] = "word",
        split_length: int = 250,
        split_overlap: int = 30,
        split_threshold: int = 0,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
        respect_sentence_boundary: bool = True,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
        encoding: str = "utf-8",
        json_content_key: str = "content",
    ) -> None:
        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.splitting_function = splitting_function
        self.respect_sentence_boundary = respect_sentence_boundary
        self.language = language
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations
        self.encoding = encoding
        self.json_content_key = json_content_key

        # initialize components
        router = FileTypeRouter(
            mime_types=[
                ConverterMimeType.CSV.value,
                ConverterMimeType.DOCX.value,
                ConverterMimeType.HTML.value,
                ConverterMimeType.JSON.value,
                ConverterMimeType.MD.value,
                ConverterMimeType.TEXT.value,
                ConverterMimeType.PDF.value,
                ConverterMimeType.PPTX.value,
                ConverterMimeType.XLSX.value,
            ]
        )

        csv = CSVToDocument(encoding=self.encoding)
        docx = DOCXToDocument()
        html = HTMLToDocument()
        json = JSONConverter(content_key=self.json_content_key)
        md = MarkdownToDocument()
        txt = TextFileToDocument(encoding=self.encoding)
        pdf = PyPDFToDocument()
        pptx = PPTXToDocument()
        xlsx = XLSXToDocument()

        joiner = DocumentJoiner()
        tabular_joiner = DocumentJoiner()

        splitter = DocumentSplitter(
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            splitting_function=self.splitting_function,
            respect_sentence_boundary=self.respect_sentence_boundary,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )

        # Create pipeline and add components
        pp = Pipeline()

        pp.add_component("router", router)

        pp.add_component("docx", docx)
        pp.add_component("html", html)
        pp.add_component("json", json)
        pp.add_component("md", md)
        pp.add_component("txt", txt)
        pp.add_component("pdf", pdf)
        pp.add_component("pptx", pptx)
        pp.add_component("xlsx", xlsx)
        pp.add_component("joiner", joiner)
        pp.add_component("splitter", splitter)
        pp.add_component("tabular_joiner", tabular_joiner)
        pp.add_component("csv", csv)


        pp.connect(f"router.{ConverterMimeType.DOCX.value}", "docx")
        pp.connect(f"router.{ConverterMimeType.HTML.value}", "html")
        pp.connect(f"router.{ConverterMimeType.JSON.value}", "json")
        pp.connect(f"router.{ConverterMimeType.MD.value}", "md")
        pp.connect(f"router.{ConverterMimeType.TEXT.value}", "txt")
        pp.connect(f"router.{ConverterMimeType.PDF.value}", "pdf")
        pp.connect(f"router.{ConverterMimeType.PPTX.value}", "pptx")
        pp.connect(f"router.{ConverterMimeType.XLSX.value}", "xlsx")

        pp.connect("joiner.documents", "splitter.documents")
        pp.connect("splitter.documents", "tabular_joiner.documents")
        pp.connect("docx.documents", "joiner.documents")
        pp.connect("html.documents", "joiner.documents")
        pp.connect("json.documents", "joiner.documents")
        pp.connect("md.documents", "joiner.documents")
        pp.connect("txt.documents", "joiner.documents")
        pp.connect("pdf.documents", "joiner.documents")
        pp.connect("pptx.documents", "joiner.documents")

        pp.connect("csv.documents", "tabular_joiner.documents")
        pp.connect("xlsx.documents", "tabular_joiner.documents")
        pp.connect(f"router.{ConverterMimeType.CSV.value}", "csv")


        output_mapping = {"tabular_joiner.documents": "documents"}
        input_mapping = {
            "sources": ["router.sources"],
            "meta": ["router.meta"]
        }

        super(AutoFileConverter, self).__init__(
            pipeline=pp,
            output_mapping=output_mapping,
            input_mapping=input_mapping
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this instance to a dictionary.
        """
        if self.splitting_function is not None:
            splitting_function = serialize_callable(self.splitting_function)
        else:
            splitting_function = self.splitting_function

        return default_to_dict(
            self,
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            splitting_function=splitting_function,
            respect_sentence_boundary=self.respect_sentence_boundary,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
            encoding=self.encoding,
            json_content_key=self.json_content_key,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoFileConverter":
        """
        Load this instance from a dictionary.
        """
        if splitting_function := data["init_parameters"].get("splitting_function"):
            data["init_parameters"]["splitting_function"] = deserialize_callable(splitting_function)

        return default_from_dict(cls, data)
