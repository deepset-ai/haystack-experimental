# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict

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
