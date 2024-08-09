# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack_experimental.dataclasses.tool import Tool, ToolInvocationError

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
