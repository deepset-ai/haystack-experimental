from typing import Any, Dict, List, Literal, Optional

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.streaming_chunk import SyncStreamingCallbackT
from haystack.utils import deserialize_callable, serialize_callable
from haystack_integrations.components.generators.anthropic.chat.chat_generator import AnthropicChatGenerator

from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.core.super_component.super_component import SuperComponent
from haystack_experimental.dataclasses.state import State, _validate_schema
from haystack_experimental.tools import Tool
from haystack_experimental.tools.tool import deserialize_tools_inplace


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


def state_to_messages_factory() -> OutputAdapter:
    """
    Factory function to create an OutputAdapter that converts state to messages.

    Returns:
        OutputAdapter: The configured OutputAdapter instance.
    """
    return OutputAdapter(template="{{ state.get('messages') }}", output_type=List[ChatMessage], unsafe=True)


def text_exit_router_factory() -> ConditionalRouter:
    """
    Factory function to create a ConditionalRouter for text exit conditions.

    Returns:
        ConditionalRouter: The configured ConditionalRouter instance.
    """
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
    return text_exit_router


def tool_exit_router_factory(exit_condition: str) -> ConditionalRouter:
    """
    Factory function to create a ConditionalRouter for tool exit conditions.

    Returns:
        ConditionalRouter: The configured ConditionalRouter instance.
    """
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
    return tool_exit_router


@component
class Agent(SuperComponent):
    """
    Agent as SuperComponent.
    """

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic"],
        model: str,
        tools: List[Tool],
        exit_condition: str = "text",
        state_schema: Optional[Dict[str, Any]] = None,
        raise_on_tool_invocation_failure: bool = False,
        streaming_callback: Optional[SyncStreamingCallbackT] = None,
        max_runs_per_component: int = 100,
        **generator_kwargs: Any,
    ) -> None:
        self._model_provider = model_provider
        self._model = model
        self._tools = tools
        self._exit_condition = exit_condition
        self._state_schema = state_schema
        self._raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self._streaming_callback = streaming_callback
        self._max_runs_per_component = max_runs_per_component
        self._generator_kwargs = generator_kwargs

        schema = state_schema or {
            "messages": {"type": List[ChatMessage]},
        }
        input_mapper = StateInputMapper(schema=schema)
        output_mapper = StateOutputMapper(schema=schema)
        loop_start = BranchJoiner(type_=State)
        loop_exit = BranchJoiner(type_=State)
        if model_provider == "openai":
            chat_generator = OpenAIChatGenerator(
                model=model, tools=tools, streaming_callback=streaming_callback, **generator_kwargs
            )
        elif model_provider == "anthropic":
            chat_generator = AnthropicChatGenerator(
                model=model, tools=tools, streaming_callback=streaming_callback, **generator_kwargs
            )
        tool_invoker = ToolInvoker(tools=tools, raise_on_failure=raise_on_tool_invocation_failure)
        state_to_msgs_tool_invoker = state_to_messages_factory()
        state_to_msgs_generator = state_to_messages_factory()
        text_exit_router = text_exit_router_factory()
        tool_exit_router = tool_exit_router_factory(exit_condition=exit_condition)

        pipeline = Pipeline(max_runs_per_component=max_runs_per_component)
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
            output_mapping={f"output_mapper.{key}": key for key in state_schema.keys()},
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this instance to a dictionary.
        """
        callback_name = serialize_callable(self._streaming_callback) if self._streaming_callback else None
        return default_to_dict(
            self,
            model_provider=self._model_provider,
            model=self._model,
            tools=[tool.to_dict() for tool in self._tools],
            exit_condition=self._exit_condition,
            state_schema=self._state_schema,
            raise_on_tool_invocation_failure=self._raise_on_tool_invocation_failure,
            streaming_callback=callback_name,
            max_runs_per_component=self._max_runs_per_component,
            generator_kwargs=self._generator_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """
        Load this instance from a dictionary.
        """
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)
