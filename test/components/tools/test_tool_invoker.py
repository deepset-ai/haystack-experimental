import json
import pytest

from haystack_experimental.dataclasses import ChatMessage, ToolCall
from haystack_experimental.dataclasses.tool import Tool, ToolInvocationError
from haystack_experimental.components.tools import ToolInvoker

# Mock tool functions
def mock_weather_tool(location):
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})

# Mock JSON schema for the tool parameters
mock_weather_tool_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"}
    },
    "required": ["location"]
}

class TestToolInvoker:

    def test_init(self):
        tool = Tool(
            name="mock_weather_tool",
            description="Provides weather information for a given location.",
            parameters=mock_weather_tool_schema,
            function=mock_weather_tool
        )
        invoker = ToolInvoker(available_tools=[tool])
        assert invoker._available_tools == [tool]
        assert invoker._tool_names == {"mock_weather_tool"}

    def test_successful_tool_call(self):
        tool = Tool(
            name="mock_weather_tool",
            description="Provides weather information for a given location.",
            parameters=mock_weather_tool_schema,
            function=mock_weather_tool
        )
        invoker = ToolInvoker(available_tools=[tool])

        tool_call = ToolCall(
            id="tool_call_1",
            tool_name="mock_weather_tool",
            arguments={"location": "Berlin"}
        )
        mock_tool_call_message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        result = invoker.run(tool_message=mock_tool_call_message)
        result_obj = json.loads(result["tool_results"][-1].tool_call_results[0].result)

        assert result_obj['weather'] == "mostly sunny"
        assert result_obj['temperature'] == 7
        assert result_obj['unit'] == "celsius"

    def test_failing_tool_call(self):
        tool = Tool(
            name="mock_weather_tool",
            description="Provides weather information for a given location.",
            parameters=mock_weather_tool_schema,
            function=mock_weather_tool
        )
        invoker = ToolInvoker(available_tools=[tool], raise_on_failure=False)

        tool_call = ToolCall(
            id="tool_call_2",
            tool_name="non_existent_tool",
            arguments={"location": "Berlin"}
        )
        mock_tool_call_message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        with pytest.raises(Exception, match="Tool non_existent_tool not found in the list of tools."):
            invoker.run(tool_message=mock_tool_call_message)

    def test_tool_invocation_error(self):
        def faulty_tool_func(location):
            raise ToolInvocationError("Tool failed to run.")

        faulty_tool_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }

        tool = Tool(
            name="faulty_tool",
            description="A tool that always fails when invoked.",
            parameters=faulty_tool_schema,
            function=faulty_tool_func
        )
        invoker = ToolInvoker(available_tools=[tool], raise_on_failure=False)

        tool_call = ToolCall(
            id="tool_call_3",
            tool_name="faulty_tool",
            arguments={"location": "Berlin"}
        )
        mock_tool_call_message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        result = invoker.run(tool_message=mock_tool_call_message)
        assert "Following error occurred while attempting to run the tool" in result["tool_results"][-1].tool_call_results[0].result

    def test_to_dict(self):
        tool = Tool(
            name="mock_weather_tool",
            description="Provides weather information for a given location.",
            parameters=mock_weather_tool_schema,
            function=mock_weather_tool
        )
        invoker = ToolInvoker(available_tools=[tool])
        data = invoker.to_dict()

        assert data == {
            "type": "haystack_experimental.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "available_tools": [tool.to_dict()],
                "raise_on_failure": True
            },
        }

    def test_from_dict(self):
        tool_data = {
            "name": "mock_weather_tool",
            "description": "Provides weather information for a given location.",
            "parameters": mock_weather_tool_schema,
            "function": "test.components.tools.test_tool_invoker.mock_weather_tool"
        }

        data = {
            "type": "haystack_experimental.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "available_tools": [tool_data],
                "raise_on_failure": True
            },
        }
        invoker = ToolInvoker.from_dict(data)
        assert invoker._tool_names == {'mock_weather_tool'}
