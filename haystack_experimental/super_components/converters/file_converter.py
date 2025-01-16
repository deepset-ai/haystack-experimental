from typing import List, Dict

from haystack.components.converters import TextFileToDocument, DOCXToDocument, XLSXToDocument, PyPDFToDocument
from haystack.components.routers import FileTypeRouter

from haystack import component, Pipeline

from haystack.core.component import Component

from haystack_experimental.components.wrappers.pipeline_wrapper import PipelineWrapper

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
            params = [param for param in sig.parameters if param != 'self']
            # Create mapping where each param maps to itself
            self.config_mapping = {param: param for param in params}



_FILE_CONVERTER_MODULES = {
    "text/plain": ComponentModule(component=TextFileToDocument),
    "application/pdf": ComponentModule(component=PyPDFToDocument),
}



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
    def __init__(self, mime_types: List[str] = None):
        if mime_types is None:
            self.mime_types = list(_FILE_CONVERTER_MODULES.keys())

        args = locals()

        pp = Pipeline()
        for mime_type in self.mime_types:
            module = _FILE_CONVERTER_MODULES[mime_type]
            config = {}
            for param, mapped_param in module.config_mapping.items():
                if param in args:
                    config[mapped_param] = args[param]

            pp.add_component(module.name, module.component(**config))

        router = FileTypeRouter(mime_types=self.mime_types)

        pp.add_component("router", router)

        for mime_type in self.mime_types:
            to_connect = _FILE_CONVERTER_MODULES[mime_type].name
            pp.connect(f"router.{mime_type}", f"{to_connect}.sources")

        super(MultiFileConverter, self).__init__(pipeline=pp)