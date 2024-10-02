# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, get_args, get_origin, get_type_hints
import inspect
from pydantic import BaseModel


from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_callable, serialize_callable

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError


class ToolInvocationError(Exception):
    """
    Exception raised when a Tool invocation fails.
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
        The callable that will be invoked when the tool is called.
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
        Convert the Tool to a dictionary.

        :returns:
            Serialized version of the Tool.
        """

        serialized = asdict(self)
        serialized["function"] = serialize_callable(self.function)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        """
        Create a Tool from a dictionary.

        :param data:
            The serialized version of the Tool.

        :returns:
            The deserialized Tool.
        """
        data["function"] = deserialize_callable(data["function"])
        return cls(**data)


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


def remove_title_key(data):
    if isinstance(data, dict):
        # Remove "title" from the current level
        if "title" in data:
            del data["title"]

        # Recursively apply to all values that are dictionaries or lists
        for key, value in list(data.items()):
            remove_title_key(value)

    elif isinstance(data, list):
        # If the current level is a list, apply the same logic to each item
        for item in data:
            remove_title_key(item)

    return data


def tool_from_function(function: Callable) -> Tool:
    from pydantic import BaseModel, create_model

    """
    Create a Tool from a function.

    :param function:
        The function to be converted to a Tool.

    :returns:
        The Tool created from the function.

    """
    print(function.__name__)
    print(function.__doc__)

    signature = inspect.signature(function)

    # Create a dictionary of fields for create_model
    fields = {}
    for name, param in signature.parameters.items():
        print(name, param)
        # If the parameter has a default value, include it in the field
        if name in ["self", "cls"]:
            # fields[name] = (param.annotation, ...)
            continue
        if param.annotation is param.empty:
            raise ValueError(f"Parameter {name} is missing a type hint in function {function.__name__}")

        # print(param.annotation.__metadata__)
        if param.default is param.empty:
            fields[name] = (param.annotation, ...)
        else:
            fields[name] = (param.annotation, param.default)

    print("FIELDS")
    print(fields)

    # # Create a Pydantic model class dynamically using __annotations__
    # model_attrs = {"__annotations__": fields}
    # # model = type(f"{function.__name__.capitalize()}Model", (BaseModel,), model_attrs)

    model = create_model(function.__name__, **fields)

    schema = remove_title_key(model.model_json_schema())
    print("SCHEMA")
    print(schema)

    # re-add back descriptions from annotations to the generated schema
    for name, param in signature.parameters.items():
        if param.annotation is not param.empty and hasattr(param.annotation, "__metadata__"):
            schema["properties"][name]["description"] = param.annotation.__metadata__[0]

    print("SCHEMA")
    print(schema)

    return Tool(name=function.__name__, description=function.__doc__, parameters=schema, function=function)

    return
    # return Tool(name=function.__name__, description=function.__doc__, parameters={}, function=function)


# def _convert_type_hints_to_json_schema(func: Callable) -> Dict:
#     type_hints = get_type_hints(func)
#     signature = inspect.signature(func)
#     required = []
#     for param_name, param in signature.parameters.items():
#         # if param.annotation == inspect.Parameter.empty:
#         #     raise TypeHintParsingException(f"Argument {param.name} is missing a type hint in function {func.__name__}")
#         if param.default == inspect.Parameter.empty:
#             required.append(param_name)

#     properties = {}
#     for param_name, param_type in type_hints.items():
#         properties[param_name] = _parse_type_hint(param_type)

#     schema = {"type": "object", "properties": properties}
#     if required:
#         schema["required"] = required

#     return schema
