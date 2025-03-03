# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.utils import serialize_callable

from haystack_experimental.components.agents import Agent
from haystack_experimental.tools import Tool, ComponentTool

import os

def weather_function(location):
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


weather_parameters = {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}


@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters=weather_parameters,
        function=weather_function,
    )

@pytest.fixture
def component_tool():
    return ComponentTool(
        name="parrot",
        description="This is a parrot.",
        component=PromptBuilder(template="{{parrot}}")
    )


class TestAgent:
    def test_serde(self, weather_tool, component_tool):
        os.environ["OPENAI_API_KEY"] = "fake-key"
        generator = OpenAIChatGenerator()
        agent = Agent(
            chat_generator=generator,
            tools=[weather_tool, component_tool],
        )

        serialized_agent = agent.to_dict()

        init_parameters = serialized_agent["init_parameters"]

        assert serialized_agent["type"] == "haystack_experimental.components.agents.agent.Agent"
        assert init_parameters["chat_generator"]["type"] == "haystack.components.generators.chat.openai.OpenAIChatGenerator"
        assert init_parameters["tools"][0]["data"]["function"] == serialize_callable(weather_function)
        assert init_parameters["tools"][1]["data"]["component"]["type"] == "haystack.components.builders.prompt_builder.PromptBuilder"

        deserialized_agent = Agent.from_dict(serialized_agent)

        assert isinstance(deserialized_agent, Agent)
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        assert deserialized_agent.tools[0].function is weather_function
        assert isinstance(deserialized_agent.tools[1]._component, PromptBuilder)


