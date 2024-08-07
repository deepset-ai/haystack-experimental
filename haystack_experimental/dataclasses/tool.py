from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict

from haystack.utils import deserialize_callable, serialize_callable
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def __post_init__(self):
        # Check that the parameters define a valid JSON schema
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as e:
            raise ValueError("The provided parameters do not define a valid JSON schema") from e

    @property
    def tool_spec(self)-> Dict[str, Any]:
        """Get the tool specification."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def invoke(self, **kwargs):
        """Invoke the tool."""
        return self.function(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        serialized = asdict(self)
        serialized["function"] = serialize_callable(self.function)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tool":
        data["function"] = deserialize_callable(data["function"])
        return cls(**data)
