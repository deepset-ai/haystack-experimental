# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack_experimental.dataclasses.tool import Tool, ToolInvocationError, deserialize_tools_inplace, _remove_title_from_schema

def get_weather_report(city: str) -> str:
    return f"Weather report for {city}: 20°C, sunny"

parameters = {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    },
    "required": ["city"]
}

class TestTool:
    def test_init(self):
        tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)

        assert tool.name == "weather"
        assert tool.description == "Get weather report"
        assert tool.parameters == parameters
        assert tool.function == get_weather_report

    def test_init_invalid_parameters(self):
        parameters = {
            "type": "invalid",
            "properties": {
                "city": {"type": "string"}
            },
        }

        with pytest.raises(ValueError):
            Tool(name="irrelevant", description="irrelevant", parameters=parameters, function=get_weather_report)

    def test_tool_spec(self):
        tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)

        assert tool.tool_spec == {"name": "weather", "description": "Get weather report", "parameters": parameters}

    def test_invoke(self):
        tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)

        assert tool.invoke(city="Berlin") == "Weather report for Berlin: 20°C, sunny"

    def test_invoke_fail(self):
        tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)

        with pytest.raises(ToolInvocationError):
            tool.invoke()

    def test_to_dict(self):
        tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)

        assert tool.to_dict() == {
            "name": "weather",
            "description": "Get weather report",
            "parameters": parameters,
            "function": "test_tool.get_weather_report"
        }

    def test_from_dict(self):
        tool_dict = {
            "name": "weather",
            "description": "Get weather report",
            "parameters": parameters,
            "function": "test_tool.get_weather_report"
        }

        tool = Tool.from_dict(tool_dict)

        assert tool.name == "weather"
        assert tool.description == "Get weather report"
        assert tool.parameters == parameters
        assert tool.function == get_weather_report


def test_deserialize_tools_inplace():
    tool = Tool(name="weather", description="Get weather report", parameters=parameters, function=get_weather_report)
    serialized_tool = tool.to_dict()
    print(serialized_tool)

    data = {"tools": [serialized_tool.copy()]}
    deserialize_tools_inplace(data)
    assert data["tools"]==[tool]

    data = {"mytools": [serialized_tool.copy()]}
    deserialize_tools_inplace(data, key="mytools")
    assert data["mytools"]==[tool]

    data = {"no_tools": 123}
    deserialize_tools_inplace(data)
    assert data=={"no_tools": 123}

def test_deserialize_tools_inplace_failures():
    data = {"key": "value"}
    deserialize_tools_inplace(data)
    assert data == {"key": "value"}

    data = {"tools": None}
    deserialize_tools_inplace(data)
    assert data == {"tools": None}

    data = {"tools": "not a list"}
    with pytest.raises(TypeError):
        deserialize_tools_inplace(data)

    data = {"tools": ["not a dictionary"]}
    with pytest.raises(TypeError):
        deserialize_tools_inplace(data)

def test_remove_title_from_schema():
    complex_schema = {'properties': {'parameter1': {'anyOf': [{'type': 'string'}, {'type': 'integer'}], 'default': 'default_value', 'title': 'Parameter1'}, 'parameter2': {'default': [1, 2, 3], 'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}, 'title': 'Parameter2', 'type': 'array'}, 'parameter3': {'anyOf': [{'type': 'string'}, {'type': 'integer'}, {'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}, 'type': 'array'}], 'default': 42, 'title': 'Parameter3'}, 'parameter4': {'anyOf': [{'type': 'string'}, {'items': {'type': 'integer'}, 'type': 'array'}, {'type': 'object'}], 'default': {'key': 'value'}, 'title': 'Parameter4'}}, 'title': 'complex_function', 'type': 'object'}

    _remove_title_from_schema(complex_schema)

    assert complex_schema == {'properties': {'parameter1': {'anyOf': [{'type': 'string'}, {'type': 'integer'}], 'default': 'default_value'}, 'parameter2': {'default': [1, 2, 3], 'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}, 'type': 'array'}, 'parameter3': {'anyOf': [{'type': 'string'}, {'type': 'integer'}, {'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}, 'type': 'array'}], 'default': 42}, 'parameter4': {'anyOf': [{'type': 'string'}, {'items': {'type': 'integer'}, 'type': 'array'}, {'type': 'object'}], 'default': {'key': 'value'}}}, 'type': 'object'}

def test_remove_title_from_schema_do_not_remove_title_property():
    """Test that the utility function only removes the 'title' keywords and not the 'title' property (if present)."""
    schema = {'properties': {'parameter1': {'type': 'string', 'title': 'Parameter1'},
                             "title": {"type": "string", "title": "Title"}},
                             'title': 'complex_function', 'type': 'object'}

    _remove_title_from_schema(schema)

    assert schema == {'properties': {'parameter1': {'type': 'string'}, "title": {"type": "string"}}, 'type': 'object'}

def test_remove_title_from_schema_handle_no_title_in_top_level():
    schema = {'properties': {'parameter1': {'type': 'string', 'title': 'Parameter1'},
                             'parameter2': {'type': 'integer', 'title': 'Parameter2'}}, 'type': 'object'}

    _remove_title_from_schema(schema)

    assert schema == {'properties': {'parameter1': {'type': 'string'}, 'parameter2': {'type': 'integer'}}, 'type': 'object'}
