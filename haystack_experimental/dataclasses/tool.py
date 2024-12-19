# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Union, get_args, get_origin, get_type_hints

from haystack import Pipeline, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_callable, serialize_callable
from pydantic import create_model

from haystack_experimental.util.utils import is_pydantic_v2_model

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
    def from_pipeline(cls, pipeline: Pipeline, name: str, description: str) -> "Tool":
        """
        Create a Tool instance from a Pipeline.

        :param pipeline:
            The Pipeline to be converted into a Tool.
        :param name:
            Name for the tool.
        :param description:
            Description of the tool.
        :returns:
            The Tool created from the Pipeline.
        :raises ValueError:
            If the pipeline is invalid or schema generation fails.
        """
        from haystack_experimental.components.tools.openai.pipeline_caller import extract_pipeline_parameters

        # Extract the parameters schema from the pipeline components
        parameters = extract_pipeline_parameters(pipeline)

        def _convert_to_dataclass(data: Any, data_type: Any) -> Any:
            """
            Recursively convert dictionaries into dataclass instances based on the provided data type.

            This function handles nested dataclasses by recursively converting each field.

            :param data:
                The input data to convert.
            :param data_type:
                The target data type, expected to be a dataclass type.
            :returns:
                An instance of the dataclass with data populated from the input dictionary.
            """
            if data is None or not isinstance(data, dict):
                return data

            # Check if the target type is a dataclass
            if is_dataclass(data_type):
                # Get field types for the dataclass (field name -> field type)
                field_types = get_type_hints(data_type)
                converted_data = {}
                # Recursively convert each field in the dataclass
                for field_name, field_type in field_types.items():
                    if field_name in data:
                        # Recursive step: convert nested dictionaries into dataclass instances
                        converted_data[field_name] = _convert_to_dataclass(data[field_name], field_type)
                # Instantiate the dataclass with the converted data
                return data_type(**converted_data)
            # If data_type is not a dataclass, return the data unchanged
            return data

        def pipeline_invoker(**kwargs):
            """
            Invokes the pipeline using keyword arguments provided by the LLM function calling/tool generated response.

            It remaps the LLM's function call payload to match the pipeline's `run` method expected format.

            :param kwargs:
                The keyword arguments to invoke the pipeline with.
            :returns:
                The result of the pipeline invocation.
            """
            pipeline_kwargs = defaultdict(dict)
            for component_param, component_input in kwargs.items():
                if "." in component_param:
                    # Split parameter into component name and parameter name
                    component_name, param_name = component_param.split(".", 1)
                    # Retrieve the component from the pipeline
                    component = pipeline.get_component(component_name)
                    # Get the parameter from the signature, checking if it exists
                    param = inspect.signature(component.run).parameters.get(param_name)
                    # Use the parameter annotation if it exists, otherwise assume a string type
                    param_type: Any = param.annotation if param else str

                    # Determine the origin type (e.g., list) and target_type (inner type)
                    origin: Any = get_origin(param_type) or param_type
                    target_type: Any
                    values_to_convert: Union[Any, List[Any]]

                    if origin is list:
                        # Parameter is a list; get the element type
                        target_type = get_args(param_type)[0]
                        values_to_convert = component_input
                    else:
                        # Parameter is a single value
                        target_type = param_type
                        values_to_convert = [component_input]

                    # Convert dictionary inputs into dataclass or Pydantic model instances if necessary
                    if is_dataclass(target_type) or is_pydantic_v2_model(target_type):
                        converted = [
                            target_type.model_validate(item)
                            if is_pydantic_v2_model(target_type)
                            else _convert_to_dataclass(item, target_type)
                            for item in values_to_convert
                            if isinstance(item, dict)
                        ]
                        # Update the component input with the converted data
                        component_input = converted if origin is list else converted[0]

                    # Map the parameter to the component in the pipeline kwargs
                    pipeline_kwargs[component_name][param_name] = component_input
                else:
                    # Handle global parameters not associated with a specific component
                    pipeline_kwargs[component_param] = component_input

            logger.debug(f"Invoking pipeline (as tool) with kwargs: {pipeline_kwargs}")
            # Invoke the pipeline with the prepared arguments
            return pipeline.run(data=pipeline_kwargs)

        # Return a new Tool instance with the pipeline invoker as the function to be called
        return Tool(name=name, description=description, parameters=parameters, function=pipeline_invoker)


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
