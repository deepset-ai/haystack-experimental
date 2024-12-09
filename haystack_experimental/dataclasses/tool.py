# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from haystack.lazy_imports import LazyImport
from haystack.logging import logging
from haystack.utils import deserialize_callable, serialize_callable
from pydantic import create_model

from haystack_experimental.dataclasses.types import OpenAPIKwargs

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

with LazyImport(message="Run 'pip install openapi-llm'") as openapi_llm_import:
    from openapi_llm.client.config import ClientConfig
    from openapi_llm.client.openapi import OpenAPIClient
    from openapi_llm.core.spec import OpenAPISpecification


class ToolInvocationError(Exception):
    """
    Exception raised when a Tool invocation fails.
    """

    pass


class SchemaGenerationError(Exception):
    """
    Exception raised when automatic schema generation fails.
    """

    pass


@dataclass
class Tool:
    """
    Data class representing a tool for which Language Models can prepare a call.

    Accurate definitions of the textual attributes such as `name` and `description`
    are important for the Language Model to correctly prepare the call.

    :param name:
        Name of the tool.
    :param description:
        Description of the tool.
    :param parameters:
        A JSON schema defining the parameters expected by the tool.
    :param function:
        The function that will be invoked when the tool is called.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def __post_init__(self):
        jsonschema_import.check()
        # Check that the parameters define a valid JSON schema
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as e:
            raise ValueError("The provided parameters do not define a valid JSON schema") from e

    @property
    def tool_spec(self) -> Dict[str, Any]:
        """
        Return the tool specification to be used by the Language Model.
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def invoke(self, **kwargs) -> Any:
        """
        Invoke the tool with the provided keyword arguments.
        """

        try:
            result = self.function(**kwargs)
        except Exception as e:
            raise ToolInvocationError(f"Failed to invoke tool `{self.name}` with parameters {kwargs}") from e
        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        serialized = asdict(self)
        serialized["function"] = serialize_callable(self.function)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """
        Deserializes the Tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized Tool.
        """
        data["function"] = deserialize_callable(data["function"])
        return cls(**data)

    @classmethod
    def from_function(cls, function: Callable, docstring_as_desc: bool = True) -> "Tool":
        """
        Create a Tool instance from a function.

        Usage example:
        ```python
        from typing import Annotated, Literal
        from haystack_experimental.dataclasses import Tool

        def get_weather(
            city: Annotated[str, "the city for which to get the weather"] = "Munich",
            unit: Annotated[Literal["Celsius", "Fahrenheit"], "the unit for the temperature"] = "Celsius"):
            '''A simple function to get the current weather for a location.'''
            return f"Weather report for {city}: 20 {unit}, sunny"

        tool = Tool.from_function(get_weather)

        print(tool)
        >>> Tool(name='get_weather', description='A simple function to get the current weather for a location.',
        >>> parameters={
        >>> 'type': 'object',
        >>> 'properties': {
        >>>     'city': {'type': 'string', 'description': 'the city for which to get the weather', 'default': 'Munich'},
        >>>     'unit': {
        >>>         'type': 'string',
        >>>         'enum': ['Celsius', 'Fahrenheit'],
        >>>         'description': 'the unit for the temperature',
        >>>         'default': 'Celsius',
        >>>     },
        >>>     }
        >>> },
        >>> function=<function get_weather at 0x7f7b3a8a9b80>)
        ```

        :param function:
            The function to be converted into a Tool.
            The function must include type hints for all parameters.
            If a parameter is annotated using `typing.Annotated`, its metadata will be used as parameter description.
        :param docstring_as_desc:
            Whether to use the function's docstring as the tool description.

        :returns:
            The Tool created from the function.

        :raises ValueError:
            If any parameter of the function lacks a type hint.
        :raises SchemaGenerationError:
            If there is an error generating the JSON schema for the Tool.
        """
        tool_description = ""
        if docstring_as_desc and function.__doc__:
            tool_description = function.__doc__

        signature = inspect.signature(function)

        # collect fields (types and defaults) and descriptions from function parameters
        fields: Dict[str, Any] = {}
        descriptions = {}

        for name, param in signature.parameters.items():
            if param.annotation is param.empty:
                raise ValueError(f"Function '{function.__name__}': parameter '{name}' does not have a type hint.")

            # if the parameter has not a default value, Pydantic requires an Ellipsis (...)
            # to explicitly indicate that the parameter is required
            default = param.default if param.default is not param.empty else ...
            fields[name] = (param.annotation, default)

            if hasattr(param.annotation, "__metadata__"):
                descriptions[name] = param.annotation.__metadata__[0]

        # create Pydantic model and generate JSON schema
        try:
            model = create_model(function.__name__, **fields)
            schema = model.model_json_schema()
        except Exception as e:
            raise SchemaGenerationError(f"Failed to create JSON schema for function '{function.__name__}'") from e

        # we don't want to include title keywords in the schema, as they contain redundant information
        # there is no programmatic way to prevent Pydantic from adding them, so we remove them later
        # see https://github.com/pydantic/pydantic/discussions/8504
        _remove_title_from_schema(schema)

        # add parameters descriptions to the schema
        for name, description in descriptions.items():
            if name in schema["properties"]:
                schema["properties"][name]["description"] = description

        return Tool(name=function.__name__, description=tool_description, parameters=schema, function=function)

    @classmethod
    def from_openapi_spec(cls, spec: Union[str, Path], **kwargs: OpenAPIKwargs) -> List["Tool"]:
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
        :raises ValueError: If the OpenAPI specification is invalid or cannot be loaded
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
            raise ValueError("spec must be a string (URL, file path, or content) or a Path object")

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

            def invoke_openapi(**kwargs):
                """
                Invoke the OpenAPI endpoint with the provided arguments.

                :param kwargs: Arguments to pass to the OpenAPI endpoint.
                :returns: Response from the OpenAPI endpoint.
                """
                return client.invoke({"name": standardized_tool_def["name"], "arguments": kwargs})

            tools.append(
                cls(
                    name=standardized_tool_def["name"],
                    description=standardized_tool_def["description"],
                    parameters=standardized_tool_def["parameters"],
                    function=invoke_openapi,
                )
            )

        return tools


def _remove_title_from_schema(schema: Dict[str, Any]):
    """
    Remove the 'title' keyword from JSON schema and contained property schemas.

    :param schema:
        The JSON schema to remove the 'title' keyword from.
    """
    schema.pop("title", None)

    for property_schema in schema["properties"].values():
        for key in list(property_schema.keys()):
            if key == "title":
                del property_schema[key]


def deserialize_tools_inplace(data: Dict[str, Any], key: str = "tools"):
    """
    Deserialize tools in a dictionary inplace.

    :param data:
        The dictionary with the serialized data.
    :param key:
        The key in the dictionary where the tools are stored.
    """
    if key in data:
        serialized_tools = data[key]

        if serialized_tools is None:
            return

        if not isinstance(serialized_tools, list):
            raise TypeError(f"The value of '{key}' is not a list")

        deserialized_tools = []
        for tool in serialized_tools:
            if not isinstance(tool, dict):
                raise TypeError(f"Serialized tool '{tool}' is not a dictionary")
            deserialized_tools.append(Tool.from_dict(tool))

        data[key] = deserialized_tools


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
                return {"name": d["name"], "description": d["description"], "parameters": schema}

        # Recurse into nested dictionaries
        for v in d.values():
            if isinstance(v, dict):
                result = _find_in_dict(v)
                if result:
                    return result
        return None

    return _find_in_dict(llm_specific_tool_def)
