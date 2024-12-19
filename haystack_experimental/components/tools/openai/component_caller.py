# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import MISSING, fields, is_dataclass
from inspect import getdoc
from typing import Any, Callable, Dict, Union, get_args, get_origin

from docstring_parser import parse
from haystack import logging
from haystack.core.component import Component
from pydantic.fields import FieldInfo

from haystack_experimental.util.utils import is_pydantic_v2_model

logger = logging.getLogger(__name__)


def extract_component_parameters(component: Component) -> Dict[str, Any]:
    """
    Extracts parameters from a Haystack component and converts them to OpenAI tools JSON format.

    :param component: The component to extract parameters from.
    :returns: A dictionary representing the component's input parameters schema.
    """
    properties = {}
    required = []

    param_descriptions = get_param_descriptions(component.run)

    for input_name, socket in component.__haystack_input__._sockets_dict.items():
        input_type = socket.type
        description = param_descriptions.get(input_name, f"Input '{input_name}' for the component.")

        try:
            property_schema = create_property_schema(input_type, description)
        except ValueError as e:
            raise ValueError(f"Error processing input '{input_name}': {e}")

        properties[input_name] = property_schema

        # Use socket.is_mandatory() to check if the input is required
        if socket.is_mandatory:
            required.append(input_name)

    parameters_schema = {"type": "object", "properties": properties}

    if required:
        parameters_schema["required"] = required

    return parameters_schema


def get_param_descriptions(method: Callable) -> Dict[str, str]:
    """
    Extracts parameter descriptions from the method's docstring using docstring_parser.

    :param method: The method to extract parameter descriptions from.
    :returns: A dictionary mapping parameter names to their descriptions.
    """
    docstring = getdoc(method)
    if not docstring:
        return {}

    parsed_doc = parse(docstring)
    param_descriptions = {}
    for param in parsed_doc.params:
        if not param.description:
            logger.warning(
                "Missing description for parameter '%s'. Please add a description in the component's "
                "run() method docstring using the format ':param %s: <description>'. "
                "This description is used to generate the Tool and helps the LLM understand how to use this parameter.",
                param.arg_name,
                param.arg_name,
            )
        param_descriptions[param.arg_name] = param.description.strip() if param.description else ""
    return param_descriptions


# ruff: noqa: PLR0912
def create_property_schema(python_type: Any, description: str, default: Any = None) -> Dict[str, Any]:
    """
    Creates a property schema for a given Python type, recursively if necessary.

    :param python_type: The Python type to create a property schema for.
    :param description: The description of the property.
    :param default: The default value of the property.
    :returns: A dictionary representing the property schema.
    """
    nullable = is_nullable_type(python_type)
    if nullable:
        non_none_types = [t for t in get_args(python_type) if t is not type(None)]
        python_type = non_none_types[0] if non_none_types else str

    origin = get_origin(python_type)
    if origin is list:
        item_type = get_args(python_type)[0] if get_args(python_type) else Any
        # recursively call create_property_schema for the item type
        items_schema = create_property_schema(item_type, "")
        items_schema.pop("description", None)
        schema = {"type": "array", "description": description, "items": items_schema}
    elif is_dataclass(python_type) or is_pydantic_v2_model(python_type):
        schema = {"type": "object", "description": description, "properties": {}}
        required_fields = []

        if is_dataclass(python_type):
            # Get the actual class if python_type is an instance otherwise use the type
            cls = python_type if isinstance(python_type, type) else python_type.__class__
            for field in fields(cls):
                field_description = f"Field '{field.name}' of '{cls.__name__}'."
                if isinstance(schema["properties"], dict):
                    schema["properties"][field.name] = create_property_schema(field.type, field_description)

        else:  # Pydantic model
            for m_name, m_field in python_type.model_fields.items():
                field_description = f"Field '{m_name}' of '{python_type.__name__}'."
                if isinstance(schema["properties"], dict):
                    schema["properties"][m_name] = create_property_schema(m_field.annotation, field_description)
                if m_field.is_required():
                    required_fields.append(m_name)

        if required_fields:
            schema["required"] = required_fields
    else:
        # Basic types
        type_mapping = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object"}
        schema = {"type": type_mapping.get(python_type, "string"), "description": description}

    if default is not None:
        schema["default"] = default

    return schema


def is_nullable_type(python_type: Any) -> bool:
    """
    Checks if the type is a Union with NoneType (i.e., Optional).

    :param python_type: The Python type to check.
    :returns: True if the type is a Union with NoneType, False otherwise.
    """
    origin = get_origin(python_type)
    if origin is Union:
        return type(None) in get_args(python_type)
    return False
