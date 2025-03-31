from typing import Any, Dict, List, Literal

from haystack import Pipeline, component
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.anthropic.chat.chat_generator import AnthropicChatGenerator

from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.core.super_component.super_component import SuperComponent
from haystack_experimental.dataclasses.state import State, _validate_schema
from haystack_experimental.tools import Tool


@component
class StateInputMapper:
    """
    Maps the input data to the state schema.
    """

    def __init__(self, schema: Dict[str, Any]):
        _validate_schema(schema)
        self.schema = schema
        for param, config in self.schema.items():
            component.set_input_type(self, name=param, type=config["type"], default=None)

    @component.output_types(state=State)
    def run(self, **kwargs) -> State:
        """
        Map the input data to the state schema.

        Returns:
            State: the mapped state object.
        """
        return {"state": State(schema=self.schema, data=kwargs)}


@component
class StateOutputMapper:
    """
    Maps the state data to the output schema.
    """

    def __init__(self, schema: Dict[str, Any]):
        _validate_schema(schema)
        self.schema = schema
        output_types = {}
        for param, config in self.schema.items():
            output_types[param] = config["type"]
        component.set_output_types(self, **output_types)

    def run(self, state: State) -> Dict[str, Any]:
        """
        Map the state data to the output schema.

        Args:
            state (State): The state object containing the data to be mapped.

        Returns:
            Dict[str, Any]: The mapped state data.
        """
        return state.data


@component
class Agent(SuperComponent):
    """
    Agent that uses a SuperComponent internally.
    """

    def __init__(
        self,
        state_schema: Dict[str, Any],
        model_provider: Literal["openai", "anthropic"],
        model: str,
        tools: List[Tool],
        exit_condition: str,
        raise_on_tool_invocation_failure: bool = False,
        **kwargs: Any,
    ) -> None:
        self.state_schema = state_schema
        input_mapper = StateInputMapper(schema=self.state_schema)
        output_mapper = StateOutputMapper(schema=self.state_schema)
        if model_provider == "openai":
            chat_generator = OpenAIChatGenerator(model=model, tools=tools)
        elif model_provider == "anthropic":
            chat_generator = AnthropicChatGenerator(model=model, tools=tools)
        tool_invoker = ToolInvoker(tools=tools, raise_on_failure=raise_on_tool_invocation_failure)
        loop_start = BranchJoiner(type_=State)
        state_to_msgs_tool_invoker = OutputAdapter(
            template="{{ state.get('messages') }}", output_type=List[ChatMessage], unsafe=True
        )
        state_to_msgs_generator = OutputAdapter(
            template="{{ state.get('messages') }}", output_type=List[ChatMessage], unsafe=True
        )
        loop_exit = BranchJoiner(type_=State)

        routes = [
            {
                "condition": "{{ llm_messages[-1].tool_call is none }}",
                "output": (
                    "{%- set _ = state.set('messages', llm_messages) if llm_messages is not undefined else None -%}"
                    "{{ state }}"
                ),
                "output_type": State,
                "output_name": "exit",
            },
            {
                "condition": "{{ True }}",  # Default route
                "output": (
                    "{%- set _ = state.set('messages', llm_messages) if llm_messages is not undefined else None -%}"
                    "{{ state }}"
                ),
                "output_type": State,
                "output_name": "continue",
            },
        ]
        text_exit_router = ConditionalRouter(routes=routes, unsafe=True)

        # Configure router conditions
        exit_condition_template = (
            f"{{%- set _ = tool_messages -%}}"
            f"{{{{ (state.get('messages')[-1].tool_call.tool_name == '{exit_condition}' "
            f"and not state.get('messages')[-1].tool_call_result.error) }}}}"
        )
        routes = [
            {
                "condition": exit_condition_template,
                "output": "{{ state }}",
                "output_type": State,
                "output_name": "exit",
            },
            {
                "condition": "{{ True }}",  # Default route
                "output": "{{ state }}",
                "output_type": State,
                "output_name": "continue",
            },
        ]
        tool_exit_router = ConditionalRouter(routes=routes, unsafe=True)

        pipeline = Pipeline(max_runs_per_component=3)
        pipeline.add_component(instance=input_mapper, name="input_mapper")
        pipeline.add_component(instance=loop_start, name="loop_start")
        pipeline.add_component(instance=state_to_msgs_generator, name="state_to_msgs_generator")
        pipeline.add_component(instance=chat_generator, name="generator")
        pipeline.add_component(instance=text_exit_router, name="text_exit_router")
        pipeline.add_component(instance=state_to_msgs_tool_invoker, name="state_to_msgs_tool_invoker")
        pipeline.add_component(instance=tool_invoker, name="tool_invoker")
        pipeline.add_component(instance=tool_exit_router, name="tool_exit_router")
        pipeline.add_component(instance=loop_exit, name="loop_exit")
        pipeline.add_component(instance=output_mapper, name="output_mapper")

        pipeline.connect("input_mapper.state", "loop_start.value")
        pipeline.connect("loop_start.value", "state_to_msgs_generator.state")
        pipeline.connect("state_to_msgs_generator.output", "generator.messages")
        pipeline.connect("generator.replies", "text_exit_router.llm_messages")
        pipeline.connect("text_exit_router.continue", "state_to_msgs_tool_invoker.state")
        pipeline.connect("state_to_msgs_tool_invoker.output", "tool_invoker.messages")
        pipeline.connect("text_exit_router.continue", "tool_invoker.state")
        pipeline.connect("loop_start.value", "text_exit_router.state")
        pipeline.connect("tool_invoker.state", "tool_exit_router.state")
        pipeline.connect("tool_invoker.tool_messages", "tool_exit_router.tool_messages")
        pipeline.connect("text_exit_router.exit", "loop_exit.value")
        pipeline.connect("tool_exit_router.exit", "loop_exit.value")
        pipeline.connect("tool_exit_router.continue", "loop_start.value")
        pipeline.connect("loop_exit.value", "output_mapper.state")

        super(Agent, self).__init__(
            pipeline=pipeline,
            input_mapping={
                "messages": ["input_mapper.messages"],
                "streaming_callback": ["generator.streaming_callback"],
            },
            output_mapping={f"output_mapper.{key}": key for key in self.state_schema.keys()},
        )
