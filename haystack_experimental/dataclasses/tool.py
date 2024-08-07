from typing import Any, Callable, Dict, Mapping, Sequence, TypedDict, Optional, List
from typing_extensions import NotRequired
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

class Property(TypedDict):
  type: str
  description: str
  enum: NotRequired[List[str]]  # `enum` is optional and can be a list of strings


class Parameters(TypedDict):
  type: str
  required: List[str]
  properties: Dict[str, Property]

class Tool:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any], function: Callable):
        self.name = name
        self.description = description

        # Validate the parameters
        try:
            Draft202012Validator.check_schema(parameters)
        except SchemaError as e:
            raise ValueError("The provided schema is invalid") from e

        self.parameters = parameters
        self.function = function

    @property
    def tool_spec(self):
        """Get the tool specification."""
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def invoke(self, **kwargs):
        """Invoke the tool."""
        return self.function(**kwargs)

    # "function": {
    #   "name": "get_current_weather",
    #   "description": "Get the current weather in a given location",
    #   "parameters": {
    #     "type": "object",
    #     "properties": {
    #       "location": {
    #         "type": "string",
    #         "description": "The city and state, e.g. San Francisco, CA",
    #       },
    #       "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    #     },
    #     "required": ["location"],
    #   },

mytool_parameters  =   {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
          },
          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "temp unit"},
        },
        "required": ["location"],
      }

mytool_name = "get_current_weather"
mytool_description = "Get the current weather in a given location"

mytool = Tool(name=mytool_name, description=mytool_description, parameters=mytool_parameters,
              function=lambda location, unit: f"Current weather in {location} is 72°F")

print(mytool.tool_spec)
print(mytool.invoke(location="San Francisco, CA", unit="fahrenheit"))

class Movie(TypedDict):
    name: str
    year: int

def print_movie_info(movie: Movie) -> None:
    print(f"{movie['name']} ({movie['year']})")

movie_data = {'name': 'Blade Runner', 'year': 1982}
print_movie_info(movie_data)
