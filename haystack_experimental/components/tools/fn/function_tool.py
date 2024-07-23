from collections import defaultdict
from collections.abc import Sequence
import dataclasses
import inspect
from typing import Any, Dict, Type, get_type_hints, get_origin, get_args, List, Union, Optional, Tuple, Callable

from pydantic import BaseModel, create_model, TypeAdapter, Field

from pydantic.json_schema import GenerateJsonSchema

from haystack_experimental.components.tools import Tool
from haystack_experimental.components.tools.openapi import LLMProvider
from haystack_experimental.components.tools.openapi._payload_extraction import create_function_payload_extractor


class NoTitleJsonSchemaGenerator(GenerateJsonSchema):
    def generate(self, schema: Dict[str, Any], mode: str = "validation") -> Dict[str, Any]:
        json_schema = super().generate(schema, mode="validation")
        self.remove_titles(json_schema)
        return json_schema

    def remove_titles(self, obj: Union[Dict, List]) -> None:
        if isinstance(obj, dict):
            obj.pop("title", None)
            for value in obj.values():
                self.remove_titles(value)
        elif isinstance(obj, list):
            for item in obj:
                self.remove_titles(item)


class FunctionTool(Tool):
    def __init__(
        self,
        llm_provider: LLMProvider,
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.llm_provider = llm_provider
        self.function = function
        self.name = name or self.function.__name__
        self.description = description or self.function.__doc__ or ""
        self.schema = self.generate_function_schema(function)

    def invoke(self, fc_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invokes the underlying function/tool with the given payload.

        :param fc_payload: The function calling payload.
        :returns: The response from the function.
        """
        try:
            # Extract and validate arguments
            args = self.get_payload_extractor()(fc_payload)
            # Call the function
            result = self.function(**args["arguments"])
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def _validate_and_extract_args(self, fc_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the given payload against the function's schema and extracts arguments.

        :param fc_payload: The payload to validate and extract.
        :returns: Validated arguments.
        """
        validated_args = {}
        for param, schema in self.schema['properties'].items():
            if param in fc_payload:
                # Handle required or optional arguments
                if schema.get('required', True) or param in fc_payload:
                    validated_args[param] = fc_payload[param]
        return validated_args

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the tools definitions used as tools LLM parameter.

        :returns: The tools definitions passed to the LLM as tools parameter.
        """
        if self.llm_provider == LLMProvider.OPENAI:
            return [{
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.schema,
                }
            }]
        elif self.llm_provider == LLMProvider.ANTHROPIC:
            return [{
                    "name": self.name,
                    "description": self.description,
                    "input_schema": self.schema,
            }]
        elif self.llm_provider == LLMProvider.COHERE:
            return [{
                "name": self.name,
                "description": self.description,
                "parameter_definitions": self.schema,
            }]
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def get_payload_extractor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the payload extractor for the LLM provider.

        This function knows how to extract the exact function payload from the LLM generated function calling payload.
        :returns: The payload extractor function.
        """
        provider_to_arguments_field_name = defaultdict(
            lambda: "arguments",
            {
                LLMProvider.ANTHROPIC: "input",
                LLMProvider.COHERE: "parameters",
            },
        )
        arguments_field_name = provider_to_arguments_field_name[self.llm_provider]
        return create_function_payload_extractor(arguments_field_name)

    def generate_schema_for_type(self, param_type: Type, is_required: Optional[bool] = True) -> Tuple[bool, Dict[str, Any]]:
        origin_type = get_origin(param_type) or param_type
        if origin_type is Union:
            args = get_args(param_type)
            if type(None) in args:
                actual_type = next(t for t in args if t is not type(None))
                return self.generate_schema_for_type(actual_type, is_required=False)
        if isinstance(origin_type, type) and issubclass(origin_type, Sequence) and origin_type is not str:
            item_type = get_args(param_type)[0]
            return is_required, {
                "type": "array",
                "items": self.generate_schema_for_type(item_type)
            }
        elif issubclass(param_type, BaseModel):
            return is_required, param_type.schema()
        elif dataclasses.is_dataclass(param_type):
            model = TypeAdapter(param_type)
            return is_required, model.json_schema()
        elif param_type in [int, float, str, bool]:
            return is_required, {"type": self.convert_primitive_type(param_type.__name__)}
        return is_required, {}

    def generate_function_schema(self, function: callable) -> Dict[str, Any]:
        sig = inspect.signature(function)
        types = get_type_hints(function)

        fields = {}
        for name, param in sig.parameters.items():
            param_type = types[name]
            is_required, schema = self.generate_schema_for_type(param_type)
            if schema:
                if is_required:
                    fields[name] = (Any, Field(json_schema_extra=schema))
                else:
                    fields[name] = (Any, Field(json_schema_extra=schema, default_factory=lambda: None))

        model = create_model(f"{function.__name__}_schema", **fields)
        return model.model_json_schema(schema_generator=NoTitleJsonSchemaGenerator)

    def convert_primitive_type(self, primitive: str) -> Any:
        if primitive == "int":
            return "integer"
        elif primitive == "float":
            return "number"
        elif primitive == "str":
            return "string"
        elif primitive == "bool":
            return "boolean"
        else:
            raise ValueError(f"Unsupported primitive type: {primitive}")