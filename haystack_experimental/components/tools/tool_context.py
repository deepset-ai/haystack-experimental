from dataclasses import dataclass
from typing import Dict, Type, Any


@dataclass
class ToolContext:
    input_schema: Dict[str, Type]
    output_schema: Dict[str, Type]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = None

    def __post_init__(self):
        # Validate input keys match schema
        if set(self.inputs.keys()) != set(self.input_schema.keys()):
            raise KeyError("Input keys don't match schema")

        # Initialize outputs with None
        self.outputs = {k: None for k in self.output_schema}

    def get_input(self, key: str) -> Any:
        if key not in self.input_schema:
            raise KeyError(f"Unknown input key: {key}")
        return self.inputs[key]

    def get_output(self, key: str) -> Any:
        if key not in self.output_schema:
            raise KeyError(f"Unknown output key: {key}")
        return self.outputs[key]

    def set_output(self, key: str, value: Any) -> None:
        if key not in self.output_schema:
            raise KeyError(f"Unknown output key: {key}")
        self.outputs[key] = value