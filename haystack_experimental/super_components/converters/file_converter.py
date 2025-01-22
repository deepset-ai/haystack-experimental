import inspect
from dataclasses import dataclass
from enum import StrEnum
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
from haystack.core.component import Component
from haystack.utils import deserialize_callable, serialize_callable

from haystack_experimental.components.wrappers.pipeline_wrapper import PipelineWrapper


@dataclass
class ComponentModule:
    component: Component
    name: str | None = None
    config_mapping: Dict[str, str] | None = None

    def __post_init__(self):
        # Set default name if not provided
        if self.name is None:
            self.name = self.component.__name__

        # Set default config mapping if not provided
        if self.config_mapping is None:
            # Get init parameters excluding self
            sig = inspect.signature(self.component.__init__)
            self.config_mapping = {param: param for param in sig.parameters if param != "self"}


class ConverterMimeType(StrEnum):
    CSV = "text/csv"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    HTML = "text/html"
    JSON = "application/json"
    MD = "text/markdown"
    TEXT = "text/plain"
    PDF = "application/pdf"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


_FILE_CONVERTER_MODULES = {
    ConverterMimeType.CSV: ComponentModule(component=CSVToDocument),
    ConverterMimeType.DOCX: ComponentModule(component=DOCXToDocument),
    ConverterMimeType.HTML: ComponentModule(component=HTMLToDocument),
    ConverterMimeType.JSON: ComponentModule(
        component=JSONConverter, config_mapping={"json_content_key": "content_key"}
    ),
    ConverterMimeType.MD: ComponentModule(component=MarkdownToDocument),
    ConverterMimeType.TEXT: ComponentModule(component=TextFileToDocument),
    ConverterMimeType.PDF: ComponentModule(component=PyPDFToDocument),
    ConverterMimeType.PPTX: ComponentModule(component=PPTXToDocument),
    ConverterMimeType.XLSX: ComponentModule(component=XLSXToDocument),
}


def _add_modules_to_pipeline(
    pipeline: Pipeline, modules: List[ComponentModule], component_args: Dict[str, Any]
) -> None:
    for module in modules:
        comp = module.component
        name = module.name
        config = {}
        for param, mapped_param in module.config_mapping.items():
            if param in component_args:
                config[mapped_param] = component_args[param]

        pipeline.add_component(name, comp(**config))


@component
class MultiFileConverter(PipelineWrapper):
    """
    A file converter that can handle multiple file types.

    Usage:
    ```
    converter = MultiFileConverter()
    converter.run(sources=["test.txt", "test.pdf"], meta={})
    ```
    """

    def __init__(
        self,
        mime_types: List[ConverterMimeType] = None,
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
        if mime_types is None:
            self.resolved_mime_types = list(_FILE_CONVERTER_MODULES.keys())
        else:
            self.resolved_mime_types = mime_types

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
        self.mime_types = mime_types

        args = locals()
        pp = Pipeline()
        converter_modules = [_FILE_CONVERTER_MODULES[mime_type] for mime_type in self.resolved_mime_types]
        _add_modules_to_pipeline(pp, converter_modules, args)

        router = FileTypeRouter(mime_types=self.resolved_mime_types)
        pp.add_component("router", router)

        joiner = DocumentJoiner()
        tabular_joiner = DocumentJoiner()
        pp.add_component("joiner", joiner)
        pp.add_component("tabular_joiner", tabular_joiner)

        for mime_type in self.resolved_mime_types:
            to_connect = _FILE_CONVERTER_MODULES[mime_type].name
            pp.connect(f"router.{mime_type}", f"{to_connect}.sources")
            if mime_type in [ConverterMimeType.XLSX, ConverterMimeType.CSV]:
                pp.connect(to_connect, "joiner")
            else:
                pp.connect(to_connect, "tabular_joiner")

        splitter_module = ComponentModule(component=DocumentSplitter, name="splitter")
        _add_modules_to_pipeline(pp, [splitter_module], args)

        pp.connect("joiner.documents", "splitter.documents")
        pp.connect("splitter.documents", "tabular_joiner.documents")

        output_mapping = {"tabular_joiner.documents": "documents"}

        super(MultiFileConverter, self).__init__(pipeline=pp, output_mapping=output_mapping)

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
            mime_types=self.mime_types,
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
    def from_dict(cls, data: Dict[str, Any]) -> "MultiFileConverter":
        """
        Load this instance from a dictionary.
        """
        if splitting_function := data["init_parameters"].get("splitting_function"):
            data["init_parameters"]["splitting_function"] = deserialize_callable(splitting_function)

        return default_from_dict(cls, data)
