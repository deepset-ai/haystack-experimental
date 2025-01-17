from typing import Any, List, Dict, Optional

from haystack.components.converters import (
    CSVToDocument,
    TextFileToDocument,
    DOCXToDocument,
    XLSXToDocument,
    PyPDFToDocument,
    PPTXToDocument,
    JSONConverter,
    HTMLToDocument,
    AzureOCRDocumentConverter,
    MarkdownToDocument
)

from haystack.components.routers import FileTypeRouter
from haystack import component, Pipeline
from haystack.core.component import Component
from haystack_experimental.components.wrappers.pipeline_wrapper import PipelineWrapper

from enum import StrEnum
from dataclasses import dataclass

import inspect

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
            self.config_mapping = {param: param for param in sig.parameters if param != 'self'}



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
    ConverterMimeType.JSON: ComponentModule(component=JSONConverter, config_mapping={"json_content_key": "content_key"}),
    ConverterMimeType.MD: ComponentModule(component=MarkdownToDocument),
    ConverterMimeType.TEXT: ComponentModule(component=TextFileToDocument),
    ConverterMimeType.PDF: ComponentModule(component=PyPDFToDocument),
    ConverterMimeType.PPTX: ComponentModule(component=PPTXToDocument),
    ConverterMimeType.XLSX: ComponentModule(component=XLSXToDocument)
}

def _add_modules_to_pipeline(pipeline: Pipeline, modules: List[ComponentModule], component_args: Dict[str, Any]) -> None:
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
    def __init__(self, mime_types: List[ComponentModule] = None, json_content_key: str = "content") -> None:
        if mime_types is None:
            self.mime_types = list(_FILE_CONVERTER_MODULES.keys())
        else:
            self.mime_types = mime_types

        args = locals()
        pp = Pipeline()
        converter_modules = [_FILE_CONVERTER_MODULES[mime_type] for mime_type in self.mime_types]
        _add_modules_to_pipeline(pp, converter_modules, args)

        router = FileTypeRouter(mime_types=self.mime_types)
        pp.add_component("router", router)

        for mime_type in self.mime_types:
            to_connect = _FILE_CONVERTER_MODULES[mime_type].name
            pp.connect(f"router.{mime_type}", f"{to_connect}.sources")

        super(MultiFileConverter, self).__init__(pipeline=pp)