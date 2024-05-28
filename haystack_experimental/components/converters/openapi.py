# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import component, logging
from haystack.dataclasses.byte_stream import ByteStream

from haystack_experimental.util.openapi import ClientConfigurationBuilder, validate_provider

logger = logging.getLogger(__name__)


@component
class OpenAPIServiceToFunctions:
    """
    Converts OpenAPI service schemas to a format suitable for OpenAI, Anthropic, or Cohere function calling.

    The definition must respect OpenAPI specification 3.0.0 or higher.
    It can be specified in JSON or YAML format.
    Each function must have:
        - unique operationId
        - description
        - requestBody and/or parameters
        - schema for the requestBody and/or parameters
    For more details on OpenAPI specification see the
    [official documentation](https://github.com/OAI/OpenAPI-Specification).

    Usage example:
    ```python
    from haystack_experimental.components.converters import OpenAPIServiceToFunctions

    converter = OpenAPIServiceToFunctions()
    result = converter.run(sources=["path/to/openapi_definition.yaml"])
    assert result["functions"]
    ```
    """

    MIN_REQUIRED_OPENAPI_SPEC_VERSION = 3

    def __init__(self, provider: Optional[str] = None):
        """
        Create an OpenAPIServiceToFunctions component.

        :param provider: The LLM provider to use, defaults to "openai".
        """
        self.llm_provider = validate_provider(provider or "openai")

    @component.output_types(functions=List[Dict[str, Any]], openapi_specs=List[Dict[str, Any]])
    def run(self, sources: List[Union[str, Path, ByteStream]]) -> Dict[str, Any]:
        """
        Converts OpenAPI definitions into LLM specific function calling format.

        :param sources:
            File paths or ByteStream objects of OpenAPI definitions (in JSON or YAML format).

        :returns:
            A dictionary with the following keys:
            - functions: Function definitions in JSON object format
            - openapi_specs: OpenAPI specs in JSON/YAML object format with resolved references

        :raises RuntimeError:
            If the OpenAPI definitions cannot be downloaded or processed.
        :raises ValueError:
            If the source type is not recognized or no functions are found in the OpenAPI definitions.
        """
        all_extracted_fc_definitions: List[Dict[str, Any]] = []
        all_openapi_specs = []

        builder = ClientConfigurationBuilder()
        for source in sources:
            source = source.to_string() if isinstance(source, ByteStream) else source
            # to get tools definitions all we need is the openapi spec
            config_openapi = builder.with_openapi_spec(source).with_provider(self.llm_provider).build()

            all_extracted_fc_definitions.extend(config_openapi.get_tools_definitions())
            all_openapi_specs.append(config_openapi.get_openapi_spec().to_dict(resolve_references=True))
        if not all_extracted_fc_definitions:
            logger.warning("No OpenAI function definitions extracted from the provided OpenAPI specification sources.")

        return {"functions": all_extracted_fc_definitions, "openapi_specs": all_openapi_specs}
