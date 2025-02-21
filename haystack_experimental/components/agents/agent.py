# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, Pipeline
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.tools import Tool
from haystack_experimental.dataclasses.state import State, _schema_from_dict, _schema_to_dict, _validate_schema



@component
class Agent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    The component processes messages and executes tools until a exit_condition condition is met.
    The exit_condition can be triggered either by a direct text response or by invoking a specific designated tool.

    ### Usage example
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.tools.tool import Tool

    tools = [Tool(name="calculator", description="..."), Tool(name="search", description="...")]
    input_types = {"search_depth": int}
    output_types = {"total_tokens": int}

    agent = Agent(
        model="anthropic:claude-3",
        generation_kwargs={"temperature": 0.7},
        tools=tools,
        exit_condition="search",
        input_variables=input_types,
        output_variables=output_types
    )

    # Run the agent
    result = agent.run(
        messages=[ChatMessage.from_user("Find information about Haystack")],
        search_depth=2
    )

    assert "messages" in result  # Contains conversation history
    assert "total_tokens" in result  # Contains tool execution outputs
    ```
    """

    def __init__( # pylint: disable=too-many-positional-arguments
        self,
        chat_generator: Union[OpenAIChatGenerator, "AnthropicChatGenerator"],
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        exit_condition: str = "text",
        state_schema: Optional[Dict[str, Any]] = None,
        max_runs_per_component: int = 100,
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
            self.pipeline.warm_up()
            self._is_warmed_up = True

    def _initialize_pipeline(self) -> None:
        """Initialize the component pipeline with all necessary components and connections."""
        joiner = BranchJoiner(type_=List[ChatMessage])
        tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False)
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

        router_output = "{%- set assistant_msg = (llm_messages[0].text|trim or 'Tool:')"\
                        "|assistant_message(none, none, llm_messages[0].tool_calls) %}"\
                        "{{ original_messages + [assistant_msg] + tool_messages }}"

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
        self.pipeline = Pipeline(max_runs_per_component=self.max_runs_per_component)
        self.pipeline.add_component(instance=self.chat_generator, name="generator")
        self.pipeline.add_component(instance=tool_invoker, name="tool_invoker")
        self.pipeline.add_component(instance=router, name="router")
        self.pipeline.add_component(instance=joiner, name="joiner")
        self.pipeline.add_component(
            instance=context_joiner, name="context_joiner"
        )

        # Connect components
        self.pipeline.connect("joiner.value", "generator.messages")
        self.pipeline.connect("generator.replies", "router.llm_messages")
        self.pipeline.connect("joiner.value", "router.original_messages")
        self.pipeline.connect("generator.replies", "tool_invoker.messages")
        self.pipeline.connect("tool_invoker.tool_messages", "router.tool_messages")
        self.pipeline.connect("router.continue", "joiner.value")
        self.pipeline.connect(
            "tool_invoker.state", "context_joiner.value"
        )
        self.pipeline.connect(
            "context_joiner.value", "tool_invoker.state"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        return default_to_dict(
            self,
            chat_generator=self.chat_generator.to_dict(),
            tools=[t.to_dict() for t in self.tools],
            system_prompt=self.system_prompt,
            exit_condition=self.exit_condition,
            state_schema=_schema_to_dict(self.state_schema),
            max_runs_per_component=self.max_runs_per_component,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized component
        """
        init_params = data.get("init_parameters", {})

        init_params["chat_generator"] = init_params["chat_generator"].from_dict()

        # Deserialize type annotations
        if "state_schema" in init_params:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if "tools" in init_params:
            init_params["tools"] = [Tool.from_dict(t) for t in init_params["tools"]]

        return default_from_dict(cls, data)

    def run(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """
        Process messages and execute tools until the exit condition is met.

        :param messages: List of chat messages to process
        :param kwargs: Additional keyword arguments matching the defined input types
        :return: Dictionary containing messages and outputs matching the defined output types
        """
        state = State(schema=self.state_schema, data=kwargs)

        if self.system_prompt is not None:
            messages = [ChatMessage.from_system(self.system_prompt)] + messages

        result = self.pipeline.run(
            data={
                "joiner": {"value": messages},
                "context_joiner": {"value": state},
                "generator": {"tools": self.tools}
            },
            include_outputs_from={"context_joiner"},
        )

        return {
            "messages": result["router"]["exit"],
            **result["context_joiner"]["value"].data,
        }
