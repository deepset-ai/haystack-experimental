# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool
from haystack.utils import Secret, deserialize_secrets_inplace

from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.core.pipeline import Pipeline
from haystack_experimental.dataclasses.state import State, _schema_from_dict, _schema_to_dict, _validate_schema

with LazyImport(message="Run 'pip install anthropic-haystack' to use Anthropic.") as anthropic_import:
    from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
        AnthropicChatGenerator,
    )

_PROVIDER_GENERATOR_MAPPING = {
    "openai": OpenAIChatGenerator,
    "anthropic": AnthropicChatGenerator,
}


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
        model: str,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        api_key: Secret = Secret.from_env_var("LLM_API_KEY", strict=False),
        exit_condition: str = "text",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        state_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent component.

        :param model: Model identifier in the format "provider:model_name"
        :param generation_kwargs: Keyword arguments for the chat model generator
        :param tools: List of Tool objects available to the agent
        :param exit_condition: Either "text" if the agent should return when it generates a message without tool calls
            or the name of a tool that will cause the agent to return once the tool was executed
        :param input_variables: Dictionary mapping input variable names to their types
        :param output_variables: Dictionary mapping output variable names to their types
        :raises ValueError: If model string format is invalid or exit_condition is not valid
        """
        if ":" not in model:
            raise ValueError("Model string must be in format 'provider:model_name'")

        provider, _ = model.split(":")
        if provider not in _PROVIDER_GENERATOR_MAPPING:
            raise ValueError(
                f"Provider must be one of {list(_PROVIDER_GENERATOR_MAPPING.keys())}"
            )


        valid_exits = ["text"] + [tool.name for tool in tools or []]
        if exit_condition not in valid_exits:
            raise ValueError(f"Exit condition must be one of {valid_exits}")

        if state_schema is not None:
            _validate_schema(state_schema)
        self.state_schema = state_schema or {}

        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_condition = exit_condition
        self.api_key = api_key

        component.set_input_type(instance=self, name="messages", type=List[ChatMessage])
        output_types = {"messages": List[ChatMessage]}
        for param, config in self.state_schema.items():
            component.set_input_type(self, name=param, type=config["type"], default=None)
            output_types[param] = config["type"]
        component.set_output_types(self, **output_types)

        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the component pipeline with all necessary components and connections."""
        provider, model_name = self.model.split(":")

        if provider == "anthropic":
            anthropic_import.check()

        # Initialize components
        generator = _PROVIDER_GENERATOR_MAPPING[provider](
            model=model_name,
            tools=self.tools,
            api_key=self.api_key,
            generation_kwargs=self.generation_kwargs,
        )
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
        self.pipeline = Pipeline()
        self.pipeline.add_component(instance=generator, name="generator")
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
        self.pipeline.connect("tool_invoker.messages", "router.tool_messages")
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
            model=self.model,
            generation_kwargs=self.generation_kwargs,
            tools=[t.to_dict() for t in self.tools],
            api_key=self.api_key.to_dict(),
            system_prompt=self.system_prompt,
            exit_condition=self.exit_condition,
            state_schema=_schema_to_dict(self.state_schema),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """
        Deserialize the component from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized component
        """
        init_params = data.get("init_parameters", {})

        # Deserialize type annotations
        if "state_schema" in init_params:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if "tools" in init_params:
            init_params["tools"] = [Tool.from_dict(t) for t in init_params["tools"]]

        deserialize_secrets_inplace(init_params, ["api_key"])

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

        self.pipeline.warm_up()

        result = self.pipeline.run(
            data={
                "joiner": {"value": messages},
                "context_joiner": {"value": state},
            },
            include_outputs_from={"context_joiner"},
        )

        return {
            "messages": result["router"]["exit"],
            **result["context_joiner"]["value"].data,
        }
