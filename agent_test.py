import os
from pathlib import Path
from typing import List

from haystack.dataclasses import ChatMessage

from haystack_experimental.super_components.agents.agent import Agent
from haystack_experimental.tools import Tool

os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your OpenAI API key


def weather_function(location: str) -> dict:
    """Dummy function to simulate a weather API call."""
    weather_info = {
        "Berlin": {"weather": "mostly sunny", "temperature": 7, "unit": "celsius"},
        "Paris": {"weather": "mostly cloudy", "temperature": 8, "unit": "celsius"},
        "Rome": {"weather": "sunny", "temperature": 14, "unit": "celsius"},
    }
    return weather_info.get(location, {"weather": "unknown", "temperature": 0, "unit": "celsius"})


weather_tool = Tool(
    name="weather_tool",
    description="Provides weather information for a given location.",
    parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
    function=weather_function,
    outputs_to_state={
        "weather": {"source": "weather"},
        "temperature": {"source": "temperature"},
        "unit": {"source": "unit"},
    },
)


#
# ==== Full SuperComponent version ======
#
tools = [weather_tool]

agent_super_component = Agent(
    state_schema={
        "messages": {"type": List[ChatMessage]},
        "weather": {"type": str},
        "temperature": {"type": int},
        "unit": {"type": str},
    },
    model_provider="openai",
    model="gpt-4o-mini",
    tools=tools,
    exit_condition="text",
    raise_on_tool_invocation_failure=False,
)
agent_super_component.pipeline.draw(Path("supercomponent_agent_pipeline.png"))


out = agent_super_component.run(
    messages=[ChatMessage.from_user("What is the weather in Berlin?")],
)
print(out)
