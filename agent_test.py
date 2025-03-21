from typing import Any, Dict, List

from haystack import Pipeline
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.agents import Agent
from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.dataclasses.state import State
from haystack_experimental.core.super_component.super_component import SuperComponent
from haystack_experimental.tools import Tool


def weather_function(location: str) -> dict:
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
)

# Agent using pipeline internally
chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
agent = Agent(chat_generator=chat_generator, tools=[weather_tool], max_runs_per_component=3)
# agent.warm_up()
# response = agent.run([ChatMessage.from_user("What is the weather in Berlin?")])


#
# ==== Full SuperComponent version ======
#
tools = [weather_tool]
raise_on_tool_invocation_failure = False

chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
joiner = BranchJoiner(type_=List[ChatMessage])
tool_invoker = ToolInvoker(tools=tools, raise_on_failure=False)
context_joiner = BranchJoiner(type_=State)
state_to_messages = OutputAdapter(
    template="{{ state.get('messages') }}",
    output_type=List[ChatMessage],
    unsafe=True
)
initialize_state = OutputAdapter(
    template="{{ schema | init_state(data={'messages': messages}) }}",
    output_type=State,
    custom_filters={"init_state": State},
    unsafe=True,
)
exit_joiner = BranchJoiner(type_=Dict[str, Any])

routes = [
    {
        "condition": "{{ llm_messages[-1].tool_call is none }}",
        "output": "{%- set _ = state.set('messages', llm_messages) if llm_messages is not undefined else None -%}{{ state.data }}",
        "output_type": Dict[str, Any],
        "output_name": "exit",
    },
    {
        "condition": "{{ True }}",  # Default route
        "output": "{%- set _ = state.set('messages', llm_messages) if llm_messages is not undefined else None -%}{{ state }}",
        "output_type": State,
        "output_name": "continue",
    },
]
router1 = ConditionalRouter(routes=routes, unsafe=True)

# Configure router conditions
exit_condition_template = "{{ (state.get('messages')[-1].tool_call.tool_name == exit_condition and not state.get('messages')[-1].tool_call_result.error) }}"
routes = [
    {
        "condition": exit_condition_template,
        "output": "{{ state.data }}",
        "output_type": Dict[str, Any],
        "output_name": "exit",
    },
    {
        "condition": "{{ True }}",  # Default route
        "output": "{{ state.get('messages') }}",
        "output_type": List[ChatMessage],
        "output_name": "continue",
    },
]
router2 = ConditionalRouter(routes=routes, unsafe=True)

pipeline = Pipeline(max_runs_per_component=3)
pipeline.add_component(instance=joiner, name="joiner")
pipeline.add_component(instance=initialize_state, name="initialize_state")
pipeline.add_component(instance=context_joiner, name="context_joiner")
pipeline.add_component(instance=chat_generator, name="generator")
pipeline.add_component(instance=router1, name="router1")
pipeline.add_component(instance=state_to_messages, name="state_to_messages")
pipeline.add_component(instance=tool_invoker, name="tool_invoker")
pipeline.add_component(instance=router2, name="router2")
pipeline.add_component(instance=exit_joiner, name="exit_joiner")

pipeline.connect("joiner.value", "generator.messages")
pipeline.connect("initialize_state.output", "context_joiner.value")
pipeline.connect("generator.replies", "router1.llm_messages")
pipeline.connect("router1.continue", "state_to_messages.state")
pipeline.connect("state_to_messages.output", "tool_invoker.messages")
pipeline.connect("router1.continue", "tool_invoker.state")
pipeline.connect("context_joiner.value", "router1.state")
pipeline.connect("tool_invoker.state", "router2.state")
pipeline.connect("router2.continue", "joiner.value")
pipeline.connect("tool_invoker.state", "context_joiner.value")
pipeline.connect("router1.exit", "exit_joiner.value")
pipeline.connect("router2.exit", "exit_joiner.value")


agent_super_component = SuperComponent(
    pipeline=pipeline,
    input_mapping={
        "messages": ["joiner.value", "initialize_state.messages"],
        "state_schema": ["initialize_state.schema"],
        "tools": ["generator.tools"],
        "exit_condition": ["router2.exit_condition"],
        "streaming_callback": ["generator.streaming_callback"],
    },
    # TODO Last issue to be solved. The output of the pipeline is a dict with key "exit"
    #      This means things like "messages" and other stuff in state are nested under "exit"
    #      Makes it hard to connect this to an AnswerBuilder without first using OutputAdapters
    #      Basically would have to manually add as many OutputAdapters as needed to flatten the dict
    output_mapping={"exit_joiner.value": "exit"}
)

out = agent_super_component.run(
    messages=[ChatMessage.from_user("What is the weather in Berlin?")],
    state_schema={},
    tools=tools,
    exit_condition="text",
)
