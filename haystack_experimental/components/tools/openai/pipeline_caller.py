# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

### This module is used to generate the OpenAI schema used for function/tool calling from a given Haystack pipeline.
### The main function is `extract_pipeline_parameters` which returns OpenAI compatible schema used in the Tool class.

from dataclasses import MISSING, fields, is_dataclass
from inspect import getdoc
from typing import Any, Callable, Dict, List, Set, Tuple, Union, get_args, get_origin

from docstring_parser import parse
from haystack import Pipeline, logging
from haystack.utils import deserialize_type

from haystack_experimental.util.utils import is_pydantic_v2_model

logger = logging.getLogger(__name__)


def extract_pipeline_parameters(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Extracts parameters from pipeline inputs and converts them to OpenAI tools JSON format.

    :param pipeline: The pipeline to extract parameters from.
    :returns: A dictionary representing the pipeline's input parameters schema.
    """
    properties = {}
    required = []

    pipeline_inputs = pipeline.inputs()

    for component_name, component_inputs in pipeline_inputs.items():
        component = pipeline.get_component(component_name)
        param_descriptions = get_param_descriptions(component.run)

        for input_name, input_info in component_inputs.items():
            # Avoid name clashes by prefixing parameter names with the component name
            prefixed_input_name = f"{component_name}.{input_name}"

            input_type = input_info.get("type") or Any

            description = param_descriptions.get(input_name, f"Input '{input_name}' for component '{component_name}'.")

            try:
                property_schema = create_property_schema(input_type, description)
            except ValueError as e:
                raise ValueError(f"Error processing input '{prefixed_input_name}': {e}")

            properties[prefixed_input_name] = property_schema

            if input_info.get("is_mandatory", False):
                required.append(prefixed_input_name)

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
    return {param.arg_name: param.description.strip() for param in parsed_doc.params}


def create_property_schema(python_type: Any, description: str, default: Any = None) -> Dict[str, Any]:  # noqa: PLR0912, PLR0915
    """
    Creates a property schema for a given Python type, recursively if necessary.

    :param python_type: The Python type to create a property schema for.
    :param description: The description of the property.
    :param default: The default value of the property.
    :returns: A dictionary representing the property schema.
    """
    nullable = is_nullable_type(python_type)
    if nullable:
        # Remove NoneType from the Union to get the actual types
        non_none_types = [t for t in get_args(python_type) if t is not type(None)]
        python_type = select_preferred_type(non_none_types)
    else:
        python_type = resolve_forward_ref(python_type)

    if not is_supported_type(python_type):
        # Assume it is a string type
        property_schema = {"type": "string", "description": description}
        if default is not None:
            property_schema["default"] = default
        return property_schema

    openai_type = get_openai_type(python_type)
    property_schema = {"type": openai_type, "description": description}

    if default is not None:
        property_schema["default"] = default

    if openai_type == "array":
        item_type = get_args(python_type)[0] if get_args(python_type) else Any
        item_type = resolve_forward_ref(item_type)
        if not is_supported_type(item_type):
            # Assume item type is string
            items_schema = {"type": "string"}
        else:
            # Create items schema without 'description'
            items_schema = create_property_schema(item_type, "")
            items_schema.pop("description", None)
        property_schema["items"] = items_schema

    elif openai_type == "object":
        # Support both dataclasses and Pydantic v2
        if is_dataclass(python_type) or is_pydantic_v2_model(python_type):
            # Handle dataclasses and Pydantic models by their fields
            property_schema["properties"] = {}
            required_fields = []

            if is_dataclass(python_type):
                model_fields = fields(python_type)
                for field in model_fields:
                    field_description = f"Field '{field.name}' of '{python_type.__name__}'."
                    field_schema = create_property_schema(field.type, field_description)
                    property_schema["properties"][field.name] = field_schema

                    # Add to required fields if the field has no default value
                    if field.default is MISSING and field.default_factory is MISSING:
                        required_fields.append(field.name)
            else:  # Pydantic v2 model
                model_fields = python_type.model_fields
                for name, field in model_fields.items():
                    field_description = f"Field '{name}' of '{python_type.__name__}'."
                    field_schema = create_property_schema(field.annotation, field_description)
                    property_schema["properties"][name] = field_schema

                    if field.is_required():
                        required_fields.append(name)

            if required_fields:
                property_schema["required"] = required_fields

        elif get_origin(python_type) is dict:
            # For dicts, specify the value type using 'additionalProperties'
            args = get_args(python_type)
            # Check for key and value type args since Dict[K, V] has 2 type parameters
            if args and len(args) == 2:  # noqa: PLR2004
                _, value_type = args
                value_type = resolve_forward_ref(value_type)
                if is_any_type(value_type):
                    # Allow any type of value
                    property_schema["additionalProperties"] = {}
                elif not is_supported_type(value_type):
                    # Assume value type is string
                    property_schema["additionalProperties"] = {"type": "string"}
                else:
                    property_schema["additionalProperties"] = create_property_schema(value_type, description)
            else:
                property_schema["additionalProperties"] = {"type": "string"}
        else:
            # Assume object is a string type
            openai_type = "string"
            property_schema = {"type": openai_type, "description": description}
            if default is not None:
                property_schema["default"] = default

    return property_schema


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


def is_basic_python_type(python_type: Any) -> bool:
    """
    Checks if the type is a basic Python type.

    :param python_type: The Python type to check.
    :returns: True if the type is a basic Python type, False otherwise.
    """
    return isinstance(python_type, type) and issubclass(python_type, (str, int, float, bool, list, dict))


def is_supported_type(python_type: Any) -> bool:
    """
    Checks if the type is a basic type, a dataclass, a Pydantic v2 model, or a supported generic type.

    :param python_type: The Python type to check.
    :returns: True if the type is a basic type, a dataclass,
    a Pydantic v2 model, or a supported generic type, False otherwise.
    """
    return (
        is_basic_python_type(python_type)
        or is_dataclass(python_type)
        or is_pydantic_v2_model(python_type)
        or is_supported_generic(python_type)
        or is_any_type(python_type)
    )


def is_supported_generic(python_type: Any) -> bool:
    """
    Checks if the type is a supported generic type like List or Dict.

    :param python_type: The Python type to check.
    :returns: True if the type is a supported generic type, False otherwise.
    """
    origin = get_origin(python_type)
    return origin in (list, List, dict, Dict)


def resolve_forward_ref(python_type: Any) -> Any:
    """
    Resolves forward references to actual types.

    :param python_type: The Python type to resolve.
    :returns: The resolved Python type.
    """
    if isinstance(python_type, str):
        python_type = deserialize_type(python_type)
    return python_type


def select_preferred_type(types: List[Any]) -> Any:
    """
    Selects the preferred type from a list of types.

    :param types: The list of types to select from.
    :returns: The preferred type.
    """
    # Resolve forward references
    types_resolved = [resolve_forward_ref(t) for t in types]

    # Prefer basic types
    for t in types_resolved:
        if is_basic_python_type(t):
            return t

    # Then prefer dataclasses
    for t in types_resolved:
        if is_dataclass(t):
            return t

    # If none matched, return the first resolved type
    if types_resolved:
        return types_resolved[0]

    raise ValueError(f"No supported types found in Union: {types}")


def get_openai_type(python_type: Any) -> str:  # noqa: PLR0911
    """
    Converts Python types to OpenAI schema types.

    :param python_type: The Python type to convert.
    :returns: The OpenAI schema type.
    """
    python_type = resolve_forward_ref(python_type)

    if is_any_type(python_type):
        return "object"  # Allow any JSON structure

    if is_basic_python_type(python_type):
        if issubclass(python_type, str):
            return "string"
        elif issubclass(python_type, int):
            return "integer"
        elif issubclass(python_type, float):
            return "number"
        elif issubclass(python_type, bool):
            return "boolean"
        elif issubclass(python_type, (list,)):
            return "array"
        elif issubclass(python_type, (dict,)):
            return "object"
    elif is_dataclass(python_type) or is_pydantic_v2_model(python_type):
        return "object"
    elif get_origin(python_type) in (list, List, tuple, Tuple, set, Set):
        return "array"
    elif get_origin(python_type) in (dict, Dict):
        return "object"

    # If none of the above conditions are met, raise an error
    raise ValueError(f"Unsupported type: {python_type}")


def is_any_type(python_type: Any) -> bool:
    """
    Checks if the type is typing.Any.

    :param python_type: The Python type to check.
    :returns: True if the type is typing.Any, False otherwise.
    """
    return python_type is Any or str(python_type) == "typing.Any"
