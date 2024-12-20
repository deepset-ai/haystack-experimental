# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

from haystack.lazy_imports import LazyImport
from haystack.logging import logging

from haystack_experimental.dataclasses import Tool

with LazyImport(message="Run 'pip install openapi-llm'") as openapi_llm_import:
    from openapi_llm.client.config import ClientConfig
    from openapi_llm.client.openapi import OpenAPIClient
    from openapi_llm.core.spec import OpenAPISpecification


logger = logging.getLogger(__name__)


class OpenAPIKwargs(TypedDict, total=False):
    """
    TypedDict for OpenAPI configuration kwargs.

    Contains all supported configuration options for Tool.from_openapi_spec()
    """

    credentials: Any  # API credentials (e.g., API key, auth token)
    request_sender: Callable[[Dict[str, Any]], Dict[str, Any]]  # Custom HTTPrequest sender function
    allowed_operations: List[str]  # A list of operations to include in the OpenAPI client.


def create_tool_from_openapi_spec(spec: Union[str, Path], operation_name: str, **kwargs: OpenAPIKwargs) -> "Tool":
    """
    Create a Tool instance from an OpenAPI specification and a specific operation name.

    :param spec: OpenAPI specification as URL, file path, or string content
    :param operation_name: Name of the operation to create a tool for
    :param kwargs: Additional configuration options for the OpenAPI client:
        - credentials: API credentials (e.g., API key, auth token)
        - request_sender: Custom callable to send HTTP requests
    :returns: Tool instance for the specified operation
    :raises ValueError: If the OpenAPI specification is invalid or cannot be loaded
    """
    # Create a new OpenAPIKwargs with the operation name
    config = OpenAPIKwargs(allowed_operations=[operation_name], credentials=kwargs.get("credentials"))

    tools = create_tools_from_openapi_spec(spec=spec, kwargs=config)
    # ensure we have only one tool
    if len(tools) != 1:
        msg = (
            f"Couldn't create a tool from OpenAPI spec '{spec}' for operation '{operation_name}'"
            "Please check that the operation name is correct and that the OpenAPI spec is valid."
        )
        raise ValueError(msg)

    return tools[0]


def create_tools_from_openapi_spec(spec: Union[str, Path], **kwargs: OpenAPIKwargs) -> List["Tool"]:
    """
    Create Tool instances from an OpenAPI specification.

    The specification can be provided as:
    - A URL pointing to an OpenAPI spec
    - A local file path to an OpenAPI spec (JSON or YAML)
    - A string containing the OpenAPI spec content (JSON or YAML)

    :param spec: OpenAPI specification as URL, file path, or string content
    :param kwargs: Additional configuration options for the OpenAPI client:
        - credentials: API credentials (e.g., API key, auth token)
        - request_sender: Custom callable to send HTTP requests
        - allowed_operations: List of operations from the OpenAPI spec to include
    :returns: List of Tool instances configured to invoke the OpenAPI service endpoints
    :raises ValueError: If the OpenAPI specification is not valid or the operation name is not found
    """
    openapi_llm_import.check()

    # Load the OpenAPI specification
    if isinstance(spec, str):
        if spec.startswith(("http://", "https://")):
            openapi_spec = OpenAPISpecification.from_url(spec)
        elif Path(spec).exists():
            openapi_spec = OpenAPISpecification.from_file(spec)
        else:
            openapi_spec = OpenAPISpecification.from_str(spec)
    elif isinstance(spec, Path):
        openapi_spec = OpenAPISpecification.from_file(str(spec))
    else:
        raise ValueError("OpenAPI spec must be a string (URL, file path, or content) or a Path object")

    # Create client configuration
    config = ClientConfig(openapi_spec=openapi_spec, **kwargs)

    # Create an OpenAPI client for invocations
    client = OpenAPIClient(config)

    # Get all tool definitions from the config
    tools = []
    for llm_specific_tool_def in config.get_tool_definitions():
        # Extract normalized tool definition
        standardized_tool_def = _standardize_tool_definition(llm_specific_tool_def)
        if not standardized_tool_def:
            logger.warning(f"Skipping {llm_specific_tool_def}, as required parameters not found")
            continue

        # Create a closure that captures the current value of standardized_tool_def
        def create_invoke_function(tool_def: Dict[str, Any]) -> Callable:
            """
            Create an invoke function with the tool definition bound to its scope.

            :param tool_def: The tool definition to bind to the invoke function.
            :returns: Function that invokes the OpenAPI endpoint.
            """

            def invoke_openapi(**kwargs):
                """
                Invoke the OpenAPI endpoint with the provided arguments.

                :param kwargs: Arguments to pass to the OpenAPI endpoint.
                :returns: Response from the OpenAPI endpoint.
                """
                return client.invoke({"name": tool_def["name"], "arguments": kwargs})

            return invoke_openapi

        tools.append(
            Tool(
                name=standardized_tool_def["name"],
                description=standardized_tool_def["description"],
                parameters=standardized_tool_def["parameters"],
                function=create_invoke_function(standardized_tool_def),
            )
        )

    return tools


def _standardize_tool_definition(llm_specific_tool_def: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Recursively extract tool parameters from different LLM provider formats.

    Supports various LLM provider formats including OpenAI, Anthropic, and Cohere.

    :param llm_specific_tool_def: Dictionary containing tool definition in provider-specific format
    :returns: Dictionary with normalized tool parameters or None if required fields not found
    """
    # Mapping of provider-specific schema field names to our Tool "parameters" field
    SCHEMA_FIELD_NAMES = [
        "parameters",  # Cohere/OpenAI
        "input_schema",  # Anthropic
        # any other field names that might contain a schema in other providers
    ]

    def _find_in_dict(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if all(k in d for k in ["name", "description"]):
            schema = None
            for field_name in SCHEMA_FIELD_NAMES:
                if field_name in d:
                    schema = d[field_name]
                    break

            if schema is not None:
                return {
                    "name": d["name"],
                    "description": d["description"],
                    "parameters": schema,
                }

        # Recurse into nested dictionaries
        for v in d.values():
            if isinstance(v, dict):
                result = _find_in_dict(v)
                if result:
                    return result
        return None

    return _find_in_dict(llm_specific_tool_def)
