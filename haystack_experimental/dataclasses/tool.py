# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional, get_args, get_origin

from haystack import logging
from haystack.core.component import Component
from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_callable, serialize_callable
from pydantic import TypeAdapter, create_model

from haystack_experimental.tools import extract_component_parameters

with LazyImport(message="Run 'pip install jsonschema'") as jsonschema_import:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError


logger = logging.getLogger(__name__)


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
    def from_function(cls, function: Callable, name: Optional[str] = None, description: Optional[str] = None) -> "Tool":
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
        :param name:
            The name of the tool. If not provided, the name of the function will be used.
        :param description:
            The description of the tool. If not provided, the docstring of the function will be used.
            To intentionally leave the description empty, pass an empty string.

        :returns:
            The Tool created from the function.

        :raises ValueError:
            If any parameter of the function lacks a type hint.
        :raises SchemaGenerationError:
            If there is an error generating the JSON schema for the Tool.
        """

        tool_description = description if description is not None else (function.__doc__ or "")

        signature = inspect.signature(function)

        # collect fields (types and defaults) and descriptions from function parameters
        fields: Dict[str, Any] = {}
        descriptions = {}

        for param_name, param in signature.parameters.items():
            if param.annotation is param.empty:
                raise ValueError(f"Function '{function.__name__}': parameter '{param_name}' does not have a type hint.")

            # if the parameter has not a default value, Pydantic requires an Ellipsis (...)
            # to explicitly indicate that the parameter is required
            default = param.default if param.default is not param.empty else ...
            fields[param_name] = (param.annotation, default)

            if hasattr(param.annotation, "__metadata__"):
                descriptions[param_name] = param.annotation.__metadata__[0]

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
        for param_name, param_description in descriptions.items():
            if param_name in schema["properties"]:
                schema["properties"][param_name]["description"] = param_description

        return Tool(name=name or function.__name__, description=tool_description, parameters=schema, function=function)

    @classmethod
    def from_component(cls, component: Component, name: str, description: str) -> "Tool":
        """
        Create a Tool instance from a Haystack component.

        :param component: The Haystack component to be converted into a Tool.
        :param name: Name for the tool.
        :param description: Description of the tool.
        :returns: The Tool created from the Component.
        :raises ValueError: If the component is invalid or schema generation fails.
        """

        if not isinstance(component, Component):
            message = (
                f"Object {component!r} is not a Haystack component. "
                "Use this method to create a Tool only with Haystack component instances."
            )
            raise ValueError(message)

        # Extract the parameters schema from the component
        parameters = extract_component_parameters(component)

        def component_invoker(**kwargs):
            """
            Invokes the component using keyword arguments provided by the LLM function calling/tool generated response.

            :param kwargs: The keyword arguments to invoke the component with.
            :returns: The result of the component invocation.
            """
            converted_kwargs = {}
            input_sockets = component.__haystack_input__._sockets_dict
            for param_name, param_value in kwargs.items():
                param_type = input_sockets[param_name].type

                # Check if the type (or list element type) has from_dict
                target_type = get_args(param_type)[0] if get_origin(param_type) is list else param_type
                if hasattr(target_type, "from_dict"):
                    if isinstance(param_value, list):
                        param_value = [target_type.from_dict(item) for item in param_value if isinstance(item, dict)]
                    elif isinstance(param_value, dict):
                        param_value = target_type.from_dict(param_value)
                else:
                    # Let TypeAdapter handle both single values and lists
                    type_adapter = TypeAdapter(param_type)
                    param_value = type_adapter.validate_python(param_value)

                converted_kwargs[param_name] = param_value
            logger.debug(f"Invoking component {type(component)} with kwargs: {converted_kwargs}")
            return component.run(**converted_kwargs)

        # Return a new Tool instance with the component invoker as the function to be called
        return Tool(name=name, description=description, parameters=parameters, function=component_invoker)


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
