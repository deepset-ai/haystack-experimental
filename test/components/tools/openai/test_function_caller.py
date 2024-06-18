# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import json
import pytest

# from haystack.utils import Secret
from haystack_experimental.components.tools import OpenAIFunctionCaller
from haystack.dataclasses import ChatMessage

WEATHER_INFO = {
    "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
    "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
    "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    "Madrid": {"weather": "sunny", "temperature": 10, "unit": "celsius"},
    "London": {"weather": "cloudy", "temperature": 9, "unit": "celsius"},
}


def mock_weather_func(location):
    if location in WEATHER_INFO:
        return WEATHER_INFO[location]
    else:
        return {"weather": "sunny", "temperature": 21.8, "unit": "fahrenheit"}

class TestOpenAIFunctionCaller:

    def test_init(self, monkeypatch):
        component = OpenAIFunctionCaller(available_functions = {"mock_weather_func": mock_weather_func})
        assert component.available_functions == {"mock_weather_func": mock_weather_func}

    def test_successful_function_call(self, monkeypatch):
        component = OpenAIFunctionCaller(available_functions = {"mock_weather_func": mock_weather_func})
        mock_assistant_message = ChatMessage.from_assistant(content='[{"id": "mock-id", "function": {"arguments": "{\\"location\\":\\"Berlin\\"}", "name": "mock_weather_func"}, "type": "function"}]',
                                                                meta={"finish_reason": "tool_calls"})
        result = component.run(messages=[mock_assistant_message])
        result_obj = json.loads(result["function_replies"][-1].content)
        assert result_obj['weather'] == WEATHER_INFO['Berlin']['weather']
        assert result_obj['temperature'] == WEATHER_INFO['Berlin']['temperature']
        assert result_obj['unit'] == WEATHER_INFO['Berlin']['unit']

    
    def test_failing_function_call(self, monkeypatch):
        component = OpenAIFunctionCaller(available_functions = {"mock_weather_func": mock_weather_func})
        mock_assistant_message = ChatMessage.from_assistant(content='[{"id": "mock-id", "function": {"arguments": "{\\"location\\":\\"Berlin\\"}", "name": "mock_weather"}, "type": "function"}]',
                                                                meta={"finish_reason": "tool_calls"})
        result = component.run(messages=[mock_assistant_message])
        assert result["function_replies"][-1].content == "I'm sorry, I tried to run a function that did not exist. Would you like me to correct it and try again?"
    
    def test_to_dict(self, monkeypatch):
        component = OpenAIFunctionCaller(available_functions = {"mock_weather_func": mock_weather_func})
        data = component.to_dict()
        assert data == {
            "type": "haystack_experimental.components.tools.openai.function_caller.OpenAIFunctionCaller",
            "init_parameters": {
                "available_functions": {'mock_weather_func': 'test.components.tools.openai.test_function_caller.mock_weather_func'}
            },
        }
    
    def test_from_dict(self, monkeypatch):
        data = {
            "type": "haystack_experimental.components.tools.openai.function_caller.OpenAIFunctionCaller",
            "init_parameters": {
                "available_functions": {'mock_weather_func': 'test.components.tools.openai.test_function_caller.mock_weather_func'},
            },
        }
        component: OpenAIFunctionCaller = OpenAIFunctionCaller.from_dict(data)
        assert component.available_functions == {'mock_weather_func': mock_weather_func}