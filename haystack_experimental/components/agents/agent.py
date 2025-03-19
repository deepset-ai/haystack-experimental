# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from haystack import Pipeline, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.core.component import Component
from haystack.core.pipeline.base import PipelineError
from haystack.core.serialization import component_from_dict
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.streaming_chunk import SyncStreamingCallbackT
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils import type_serialization

from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.dataclasses.state import State, _schema_from_dict, _schema_to_dict, _validate_schema
from haystack_experimental.tools import Tool, deserialize_tools_inplace

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haystack_integrations.components.generators.anthropic.chat.chat_generator import AnthropicChatGenerator


@component
class Agent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    The component processes messages and executes tools until a exit_condition condition is met.
    The exit_condition can be triggered either by a direct text response or by invoking a specific designated tool.

    ### Usage example
    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools.tool import Tool
    from haystack_experimental.components.agents import Agent

    tools = [Tool(name="calculator", description="..."), Tool(name="search", description="...")]

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=tools,
        exit_condition="search",
    )

    # Run the agent
    result = agent.run(
        messages=[ChatMessage.from_user("Find information about Haystack")]
    )

    assert "messages" in result  # Contains conversation history
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: Union[OpenAIChatGenerator, "AnthropicChatGenerator"],
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        exit_condition: str = "text",
        state_schema: Optional[Dict[str, Any]] = None,
        max_runs_per_component: int = 100,
        raise_on_tool_invocation_failure: bool = False,
        streaming_callback: Optional[SyncStreamingCallbackT] = None,
    ):
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use.
        :param tools: List of Tool objects available to the agent
        :param system_prompt: System prompt for the agent.
        :param exit_condition: Either "text" if the agent should return when it generates a message without tool calls
            or the name of a tool that will cause the agent to return once the tool was executed
        :param state_schema: The schema for the runtime state used by the tools.
        :param max_runs_per_component: Maximum number of runs per component. Agent will raise an exception if a
            component exceeds the maximum number of runs per component.
        :param raise_on_tool_invocation_failure: Should the agent raise an exception when a tool invocation fails?
            If set to False, the exception will be turned into a chat message and passed to the LLM.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        """
        valid_exits = ["text"] + [tool.name for tool in tools or []]
        if exit_condition not in valid_exits:
            raise ValueError(f"Exit condition must be one of {valid_exits}")

        if state_schema is not None:
            _validate_schema(state_schema)
        self.state_schema = state_schema or {}

        self.chat_generator = chat_generator
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_condition = exit_condition
        self.max_runs_per_component = max_runs_per_component
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback

        output_types = {"messages": List[ChatMessage]}
        for param, config in self.state_schema.items():
            component.set_input_type(self, name=param, type=config["type"], default=None)
            output_types[param] = config["type"]
        component.set_output_types(self, **output_types)

        self._initialize_pipeline()

        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the Agent.
        """
        if not self._is_warmed_up:
            self._pipeline.warm_up()
            self._is_warmed_up = True

    def _initialize_pipeline(self) -> None:
        """Initialize the component pipeline with all necessary components and connections."""
        joiner = BranchJoiner(type_=List[ChatMessage])
        tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=self.raise_on_tool_invocation_failure)
        context_joiner = BranchJoiner(type_=State)

        # Configure router conditions
        if self.exit_condition == "text":
            exit_condition_template = "{{ llm_messages[0].tool_call is none }}"
        else:
            exit_condition_template = (
                "{{ llm_messages[0].tool_call is none or (llm_messages[0].tool_call.tool_name == '"
                + self.exit_condition
                + "' and not tool_messages[0].tool_call_result.error) }}"
            )

        router_output = (
            "{%- set assistant_msg = (llm_messages[0].text|trim or 'Tool:')"
            "|assistant_message(none, none, llm_messages[0].tool_calls) %}"
            "{{ original_messages + [assistant_msg] + tool_messages }}"
        )

        routes = [
            {
                "condition": exit_condition_template,
                "output": router_output,
                "output_type": List[ChatMessage],
                "output_name": "exit",
            },
            {
                "condition": "{{ True }}",  # Default route
                "output": router_output,
                "output_type": List[ChatMessage],
                "output_name": "continue",
            },
        ]

        router = ConditionalRouter(
            routes=routes,
            custom_filters={"assistant_message": ChatMessage.from_assistant},
            unsafe=True,
        )

        # Set up pipeline
        self._pipeline = Pipeline(max_runs_per_component=self.max_runs_per_component)
        self._pipeline.add_component(instance=self.chat_generator, name="generator")
        self._pipeline.add_component(instance=tool_invoker, name="tool_invoker")
        self._pipeline.add_component(instance=router, name="router")
        self._pipeline.add_component(instance=joiner, name="joiner")
        self._pipeline.add_component(instance=context_joiner, name="context_joiner")

        # Connect components
        self._pipeline.connect("joiner.value", "generator.messages")
        self._pipeline.connect("generator.replies", "router.llm_messages")
        self._pipeline.connect("joiner.value", "router.original_messages")
        self._pipeline.connect("generator.replies", "tool_invoker.messages")
        self._pipeline.connect("tool_invoker.tool_messages", "router.tool_messages")
        self._pipeline.connect("router.continue", "joiner.value")
        self._pipeline.connect("tool_invoker.state", "context_joiner.value")
        self._pipeline.connect("context_joiner.value", "tool_invoker.state")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        if self.streaming_callback is not None:
            streaming_callback = serialize_callable(self.streaming_callback)
        else:
            streaming_callback = None

        return default_to_dict(
            self,
            chat_generator=self.chat_generator.to_dict(),
            tools=[t.to_dict() for t in self.tools],
            system_prompt=self.system_prompt,
            exit_condition=self.exit_condition,
            state_schema=_schema_to_dict(self.state_schema),
            max_runs_per_component=self.max_runs_per_component,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            streaming_callback=streaming_callback
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """
        Deserialize the agent from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized agent
        """
        init_params = data.get("init_parameters", {})

        init_params["chat_generator"] = Agent._load_component(init_params["chat_generator"])

        if "state_schema" in init_params:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_inplace(init_params, key="tools")

        return default_from_dict(cls, data)

    @staticmethod
    def _load_component(component_data: Dict[str, Any]) -> Component:
        if component_data["type"] not in component.registry:
            try:
                # Import the module first...
                module, _ = component_data["type"].rsplit(".", 1)
                logger.debug("Trying to import module {module_name}", module_name=module)
                type_serialization.thread_safe_import(module)
                # ...then try again
                if component_data["type"] not in component.registry:
                    raise PipelineError(
                        f"Successfully imported module {module} but can't find it in the component registry."
                        "This is unexpected and most likely a bug."
                    )
            except (ImportError, PipelineError) as e:
                raise PipelineError(f"Component '{component_data['type']}' not imported.") from e

        # Create a new one
        component_class = component.registry[component_data["type"]]
        instance = component_from_dict(component_class, component_data, "")
        return instance

    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[SyncStreamingCallbackT] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process messages and execute tools until the exit condition is met.

        :param messages: List of chat messages to process
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :return: Dictionary containing messages and outputs matching the defined output types
        """
        if not self._is_warmed_up:
            raise RuntimeError("The component Agent wasn't warmed up. Run 'warm_up()' before calling 'run()'.")

        state = State(schema=self.state_schema, data=kwargs)

        if self.system_prompt is not None:
            messages = [ChatMessage.from_system(self.system_prompt)] + messages

        generator_inputs = {"tools": self.tools}

        selected_callback = streaming_callback or self.streaming_callback
        if selected_callback is not None:
            generator_inputs["streaming_callback"] = selected_callback

        result = self._pipeline.run(
            data={
                "joiner": {"value": messages},
                "context_joiner": {"value": state},
                "generator": generator_inputs,
            },
            include_outputs_from={"context_joiner"},
        )

        return {"messages": result["router"]["exit"], **result["context_joiner"]["value"].data}
