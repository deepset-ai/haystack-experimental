# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Callable, Dict, List, Optional

# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import jsonref

MIN_REQUIRED_OPENAPI_SPEC_VERSION = 3

logger = logging.getLogger(__name__)


def openai_converter(schema: "OpenAPISpecification") -> List[Dict[str, Any]]:  # noqa: F821
    """
    Converts OpenAPI specification to a list of function suitable for OpenAI LLM function calling.

    :param schema: The OpenAPI specification to convert.
    :return: A list of dictionaries, each representing a function definition.
    """
    resolved_schema = jsonref.replace_refs(schema.spec_dict)
    fn_definitions = _openapi_to_functions(
        resolved_schema, "parameters", _parse_endpoint_spec_openai
    )
    return [{"type": "function", "function": fn} for fn in fn_definitions]


def anthropic_converter(schema: "OpenAPISpecification") -> List[Dict[str, Any]]:  # noqa: F821
    """
    Converts an OpenAPI specification to a list of function definitions for Anthropic LLM function calling.

    :param schema: The OpenAPI specification to convert.
    :return: A list of dictionaries, each representing a function definition.
    """
    resolved_schema = jsonref.replace_refs(schema.spec_dict)
    return _openapi_to_functions(
        resolved_schema, "input_schema", _parse_endpoint_spec_openai
    )


def cohere_converter(schema: "OpenAPISpecification") -> List[Dict[str, Any]]:  # noqa: F821
    """
    Converts an OpenAPI specification to a list of function definitions for Cohere LLM function calling.

    :param schema: The OpenAPI specification to convert.
    :return: A list of dictionaries, each representing a function definition.
    """
    resolved_schema = jsonref.replace_refs(schema.spec_dict)
    return _openapi_to_functions(
        resolved_schema, "not important for cohere", _parse_endpoint_spec_cohere
    )


def _openapi_to_functions(
    service_openapi_spec: Dict[str, Any],
    parameters_name: str,
    parse_endpoint_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extracts functions from the OpenAPI specification, converts them into a function schema.
    """

    # Doesn't enforce rigid spec validation because that would require a lot of dependencies
    # We check the version and require minimal fields to be present, so we can extract functions
    spec_version = service_openapi_spec.get("openapi")
    if not spec_version:
        raise ValueError(
            f"Invalid OpenAPI spec provided. Could not extract version from {service_openapi_spec}"
        )
    service_openapi_spec_version = int(spec_version.split(".")[0])
    # Compare the versions
    if service_openapi_spec_version < MIN_REQUIRED_OPENAPI_SPEC_VERSION:
        raise ValueError(
            f"Invalid OpenAPI spec version {service_openapi_spec_version}. Must be "
            f"at least {MIN_REQUIRED_OPENAPI_SPEC_VERSION}."
        )
    functions: List[Dict[str, Any]] = []
    for paths in service_openapi_spec["paths"].values():
        for path_spec in paths.values():
            function_dict = parse_endpoint_fn(path_spec, parameters_name)
            if function_dict:
                functions.append(function_dict)
    return functions


def _parse_endpoint_spec_openai(
    resolved_spec: Dict[str, Any], parameters_name: str
) -> Dict[str, Any]:
    """
    Parses an OpenAPI endpoint specification for OpenAI.
    """
    if not isinstance(resolved_spec, dict):
        logger.warning(
            "Invalid OpenAPI spec format provided. Could not extract function."
        )
        return {}
    function_name = resolved_spec.get("operationId")
    description = resolved_spec.get("description") or resolved_spec.get("summary", "")
    schema: Dict[str, Any] = {"type": "object", "properties": {}}
    # requestBody section
    req_body_schema = (
        resolved_spec.get("requestBody", {})
        .get("content", {})
        .get("application/json", {})
        .get("schema", {})
    )
    if "properties" in req_body_schema:
        for prop_name, prop_schema in req_body_schema["properties"].items():
            schema["properties"][prop_name] = _parse_property_attributes(prop_schema)
        if "required" in req_body_schema:
            schema.setdefault("required", []).extend(req_body_schema["required"])

    # parameters section
    for param in resolved_spec.get("parameters", []):
        if "schema" in param:
            schema_dict = _parse_property_attributes(param["schema"])
            # these attributes are not in param[schema] level but on param level
            useful_attributes = ["description", "pattern", "enum"]
            schema_dict.update(
                {key: param[key] for key in useful_attributes if param.get(key)}
            )
            schema["properties"][param["name"]] = schema_dict
            if param.get("required", False):
                schema.setdefault("required", []).append(param["name"])

    if function_name and description and schema["properties"]:
        return {
            "name": function_name,
            "description": description,
            parameters_name: schema,
        }
    logger.warning(
        "Invalid OpenAPI spec format provided. Could not extract function from %s",
        resolved_spec,
    )
    return {}


def _parse_property_attributes(
    property_schema: Dict[str, Any], include_attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Recursively parses the attributes of a property schema.
    """
    include_attributes = include_attributes or ["description", "pattern", "enum"]
    schema_type = property_schema.get("type")
    parsed_schema = {"type": schema_type} if schema_type else {}
    for attr in include_attributes:
        if attr in property_schema:
            parsed_schema[attr] = property_schema[attr]
    if schema_type == "object":
        properties = property_schema.get("properties", {})
        parsed_properties = {
            prop_name: _parse_property_attributes(prop, include_attributes)
            for prop_name, prop in properties.items()
        }
        parsed_schema["properties"] = parsed_properties
        if "required" in property_schema:
            parsed_schema["required"] = property_schema["required"]
    elif schema_type == "array":
        items = property_schema.get("items", {})
        parsed_schema["items"] = _parse_property_attributes(items, include_attributes)
    return parsed_schema


def _parse_endpoint_spec_cohere(
    operation: Dict[str, Any], ignored_param: str
) -> Dict[str, Any]:
    """
    Parses an endpoint specification for Cohere.
    """
    function_name = operation.get("operationId")
    description = operation.get("description") or operation.get("summary", "")
    parameter_definitions = _parse_parameters(operation)
    if function_name:
        return {
            "name": function_name,
            "description": description,
            "parameter_definitions": parameter_definitions,
        }
    logger.warning("Operation missing operationId, cannot create function definition.")
    return {}


def _parse_parameters(operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the parameters from an operation specification.
    """
    parameters = {}
    for param in operation.get("parameters", []):
        if "schema" in param:
            parameters[param["name"]] = _parse_schema(
                param["schema"],
                param.get("required", False),
                param.get("description", ""),
            )
    if "requestBody" in operation:
        content = (
            operation["requestBody"].get("content", {}).get("application/json", {})
        )
        if "schema" in content:
            schema_properties = content["schema"].get("properties", {})
            required_properties = content["schema"].get("required", [])
            for name, schema in schema_properties.items():
                parameters[name] = _parse_schema(
                    schema, name in required_properties, schema.get("description", "")
                )
    return parameters


def _parse_schema(
    schema: Dict[str, Any], required: bool, description: str
) -> Dict[str, Any]:  # noqa: FBT001
    """
    Parses a schema part of an operation specification.
    """
    schema_type = _get_type(schema)
    if schema_type == "object":
        # Recursive call for complex types
        properties = schema.get("properties", {})
        nested_parameters = {
            name: _parse_schema(
                schema=prop_schema,
                required=bool(name in schema.get("required", False)),
                description=prop_schema.get("description", ""),
            )
            for name, prop_schema in properties.items()
        }
        return {
            "type": schema_type,
            "description": description,
            "properties": nested_parameters,
            "required": required,
        }
    return {"type": schema_type, "description": description, "required": required}


def _get_type(schema: Dict[str, Any]) -> str:
    type_mapping = {
        "integer": "int",
        "string": "str",
        "boolean": "bool",
        "number": "float",
        "object": "object",
        "array": "list",
    }
    schema_type = schema.get("type", "object")
    if schema_type not in type_mapping:
        raise ValueError(f"Unsupported schema type {schema_type}")
    return type_mapping[schema_type]
