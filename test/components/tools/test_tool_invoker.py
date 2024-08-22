import json
import pytest
import datetime

from haystack import Pipeline

from haystack_experimental.dataclasses import ChatMessage, ToolCall, ToolCallResult, ChatRole
from haystack_experimental.dataclasses.tool import Tool, ToolInvocationError
from haystack_experimental.components.tools.tool_invoker import ToolInvoker, ToolNotFoundException
from haystack_experimental.components.generators.chat import OpenAIChatGenerator


def weather_function(location):
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


weather_parameters = {
    "type": "object",
    "properties": {
        "location": {"type": "string"}
    },
    "required": ["location"]
}

@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters=weather_parameters,
        function=weather_function
    )

@pytest.fixture
def faulty_tool():
    def faulty_tool_func(location):
        raise Exception("This tool always fails.")

    faulty_tool_parameters = {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"]
    }

    return Tool(
        name="faulty_tool",
        description="A tool that always fails when invoked.",
        parameters=faulty_tool_parameters,
        function=faulty_tool_func
    )

@pytest.fixture
def invoker(weather_tool):
    return ToolInvoker(tools=[weather_tool], raise_on_failure=True)

@pytest.fixture
def faulty_invoker(faulty_tool):
    return ToolInvoker(tools=[faulty_tool], raise_on_failure=True)

class TestToolInvoker:

    def test_init(self, weather_tool):
        invoker = ToolInvoker(tools=[weather_tool])

        assert invoker.tools == [weather_tool]
        assert invoker._tools_with_names == {'weather_tool': weather_tool}
        assert invoker.raise_on_failure

    def test_init_fails_wo_tools(self):
        with pytest.raises(ValueError):
            ToolInvoker(tools=[])

    def test_init_fails_with_duplicate_tool_names(self, weather_tool, faulty_tool):
        with pytest.raises(ValueError):
            ToolInvoker(tools=[weather_tool, weather_tool])

        new_tool = faulty_tool
        new_tool.name = "weather_tool"
        with pytest.raises(ValueError):
            ToolInvoker(tools=[weather_tool, new_tool])

    def test_convert_tool_result_to_string(self):
        result = {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"}
        result_str = ToolInvoker._convert_tool_result_to_string(result)
        assert result_str == json.dumps(result)

        result = datetime.datetime(2022, 1, 1)
        result_str = ToolInvoker._convert_tool_result_to_string(result)
        assert result_str == str(result)


    def test_run(self, invoker):

        tool_call = ToolCall(
            tool_name="weather_tool",
            arguments={"location": "Berlin"}
        )
        message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        result = invoker.run(message=message)
        assert "tool_messages" in result
        assert len(result["tool_messages"]) == 1

        tool_message = result["tool_messages"][0]
        assert isinstance(tool_message, ChatMessage)
        assert tool_message.is_from(ChatRole.TOOL)

        assert tool_message.tool_call_results
        tool_call_result = tool_message.tool_call_result

        assert isinstance(tool_call_result, ToolCallResult)
        assert tool_call_result.result == json.dumps({"weather": "mostly sunny", "temperature": 7, "unit": "celsius"})
        assert tool_call_result.origin == tool_call

    def test_run_with_invalid_message(self, invoker):
        message_from_user = ChatMessage.from_user(text="Message from user.")
        with pytest.raises(ValueError):
            invoker.run(message=message_from_user)

        message_wo_tool_calls = ChatMessage.from_assistant(text="Message without tool calls.")
        with pytest.raises(ValueError):
            invoker.run(message=message_wo_tool_calls)

    def test_tool_not_found_error(self, invoker):
        tool_call = ToolCall(
            tool_name="non_existent_tool",
            arguments={"location": "Berlin"}
        )
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        with pytest.raises(ToolNotFoundException):
            invoker.run(message=tool_call_message)

    def test_tool_not_found_does_not_raise_exception(self, invoker):
        invoker.raise_on_failure = False

        tool_call = ToolCall(
            tool_name="non_existent_tool",
            arguments={"location": "Berlin"}
        )
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        result = invoker.run(message=tool_call_message)
        tool_message = result["tool_messages"][0]
        assert "not found" in tool_message.tool_call_results[0].result

    def test_tool_invocation_error(self, faulty_invoker):
        tool_call = ToolCall(
            tool_name="faulty_tool",
            arguments={"location": "Berlin"}
        )
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        with pytest.raises(ToolInvocationError):
            faulty_invoker.run(message=tool_call_message)

    def test_tool_invocation_error_does_not_raise_exception(self, faulty_invoker):
        faulty_invoker.raise_on_failure = False

        tool_call = ToolCall(
            tool_name="faulty_tool",
            arguments={"location": "Berlin"}
        )
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[tool_call]
        )

        result = faulty_invoker.run(message=tool_call_message)
        tool_message = result["tool_messages"][0]
        assert "invocation failed" in tool_message.tool_call_results[0].result


    def test_to_dict(self, invoker, weather_tool):
        data = invoker.to_dict()
        assert data == {
            "type": "haystack_experimental.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "tools": [weather_tool.to_dict()],
                "raise_on_failure": True
            },
        }

    def test_from_dict(self, weather_tool):
        data = {
            "type": "haystack_experimental.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
                "tools": [weather_tool.to_dict()],
                "raise_on_failure": True
            },
        }
        invoker = ToolInvoker.from_dict(data)
        assert invoker.tools == [weather_tool]
        assert invoker._tools_with_names == {'weather_tool': weather_tool}
        assert invoker.raise_on_failure

    def test_serde_in_pipeline(self, invoker, weather_tool, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        pipeline = Pipeline()
        pipeline.add_component("invoker", invoker)
        pipeline.add_component("chatgenerator", OpenAIChatGenerator())
        pipeline.connect("invoker", "chatgenerator")

        pipeline_dict = pipeline.to_dict()
        assert pipeline_dict == {'metadata': {}, 'max_loops_allowed': 100, 'components': {'invoker': {'type': 'haystack_experimental.components.tools.tool_invoker.ToolInvoker', 'init_parameters': {'tools': [{'name': 'weather_tool', 'description': 'Provides weather information for a given location.', 'parameters': {'type': 'object', 'properties': {'location': {'type': 'string'}}, 'required': ['location']}, 'function': 'test.components.tools.test_tool_invoker.weather_function'}], 'raise_on_failure': True}}, 'chatgenerator': {'type': 'haystack_experimental.components.generators.chat.openai.OpenAIChatGenerator', 'init_parameters': {'model': 'gpt-3.5-turbo', 'streaming_callback': None, 'api_base_url': None, 'organization': None, 'generation_kwargs': {}, 'api_key': {'type': 'env_var', 'env_vars': ['OPENAI_API_KEY'], 'strict': True}, 'tools': None, 'tools_strict': False}}}, 'connections': [{'sender': 'invoker.tool_messages', 'receiver': 'chatgenerator.messages'}]}

        new_pipeline = Pipeline.from_dict(pipeline_dict)

        assert new_pipeline==pipeline
