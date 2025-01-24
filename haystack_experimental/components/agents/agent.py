# type: ignore
from typing import Dict, Any, List, Optional, Type

from haystack import component, default_from_dict, default_to_dict
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.dataclasses import ChatMessage
from haystack.utils.type_serialization import serialize_type, deserialize_type

from haystack.tools import Tool

from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
    AnthropicChatGenerator,
)
from haystack.components.generators.chat.openai import OpenAIChatGenerator

from haystack_experimental.core.pipeline import Pipeline
from haystack_experimental.components.tools import ToolInvoker
from haystack_experimental.components.tools.tool_context import ToolContext

_PROVIDER_GENERATOR_MAPPING = {
    "openai": OpenAIChatGenerator,
    "anthropic": AnthropicChatGenerator,
}


@component
class Agent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    The component processes messages and executes tools until a handoff condition is met. The handoff can be triggered
    either by a direct text response or by invoking a specific designated tool.

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
        handoff="search",
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

    def __init__(
        self,
        model: str,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        api_key: Secret = Secret.from_env_var("LLM_API_KEY", strict=False),
        handoff: str = "text",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        input_variables: Optional[Dict[str, Type]] = None,
        output_variables: Optional[Dict[str, Type]] = None,
    ):
        """
        Initialize the agent component.

        :param model: Model identifier in the format "provider:model_name"
        :param generation_kwargs: Keyword arguments for the chat model generator
        :param tools: List of Tool objects available to the agent
        :param handoff: Either "text" if the agent should return when it generates a message without tool calls
            or the name of a tool that will cause the agent to return once the tool was executed
        :param input_variables: Dictionary mapping input variable names to their types
        :param output_variables: Dictionary mapping output variable names to their types
        :raises ValueError: If model string format is invalid or handoff is not valid
        """
        if ":" not in model:
            raise ValueError("Model string must be in format 'provider:model_name'")

        provider, model_name = model.split(":")
        if provider not in _PROVIDER_GENERATOR_MAPPING:
            raise ValueError(
                f"Provider must be one of {list(_PROVIDER_GENERATOR_MAPPING.keys())}"
            )

        valid_handoffs = ["text"] + [tool.name for tool in tools]
        if handoff not in valid_handoffs:
            raise ValueError(f"Handoff must be one of {valid_handoffs}")

        # Store instance variables
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.handoff = handoff
        self.input_variables = input_variables or {}
        self.output_variables = output_variables or {}
        self.api_key = api_key

        # Set input/output types
        input_types = {"messages": List[ChatMessage]}
        input_types.update(self.input_variables)
        component.set_input_types(instance=self, **input_types)

        output_types = {"messages": List[ChatMessage]}
        output_types.update(self.output_variables)
        component.set_output_types(instance=self, **output_types)

        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the component pipeline with all necessary components and connections."""
        provider, model_name = self.model.split(":")

        # Initialize components
        generator = _PROVIDER_GENERATOR_MAPPING[provider](
            model=model_name,
            tools=self.tools,
            api_key=self.api_key,
            generation_kwargs=self.generation_kwargs,
        )
        joiner = BranchJoiner(type_=List[ChatMessage])
        tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False)
        context_joiner = BranchJoiner(type_=ToolContext)

        # Configure router conditions
        if self.handoff == "text":
            handoff_condition = "{{ llm_messages[0].tool_call is none }}"
        else:
            handoff_condition = (
                "{{ llm_messages[0].tool_call is none or (llm_messages[0].tool_call.tool_name == '"
                + self.handoff
                + "' and not tool_messages[0].tool_call_result.error) }}"
            )

        router_output = "{%- set assistant_msg = (llm_messages[0].text|trim or 'Tool:')|assistant_message(none, none, llm_messages[0].tool_calls) %}{{ original_messages + [assistant_msg] + tool_messages }}"

        routes = [
            {
                "condition": handoff_condition,
                "output": router_output,
                "output_type": List[ChatMessage],
                "output_name": "handoff",
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
            "tool_invoker.context", "context_joiner.value"
        )
        self.pipeline.connect(
            "context_joiner.value", "tool_invoker.context"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        # Serialize type annotations
        serialized_context_variables = {
            name: serialize_type(type_)
            for name, type_ in self.input_variables.items()
        }
        serialized_output_variables = {
            name: serialize_type(type_) for name, type_ in self.output_variables.items()
        }

        serialized_tools = [t.to_dict() for t in self.tools]

        return default_to_dict(
            self,
            model=self.model,
            generation_kwargs=self.generation_kwargs,
            tools=serialized_tools,
            api_key=self.api_key.to_dict(),
            system_prompt=self.system_prompt,
            handoff=self.handoff,
            context_variables=serialized_context_variables,
            output_variables=serialized_output_variables,
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
        if "input_variables" in init_params:
            init_params["input_variables"] = {
                name: deserialize_type(type_)
                for name, type_ in init_params["input_variables"].items()
            }
        if "output_variables" in init_params:
            init_params["output_variables"] = {
                name: deserialize_type(type_)
                for name, type_ in init_params["output_variables"].items()
            }

        if "tools" in init_params:
            init_params["tools"] = [Tool.from_dict(t) for t in init_params["tools"]]

        deserialize_secrets_inplace(init_params, ["api_key"])

        return default_from_dict(cls, data)

    def run(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """
        Process messages and execute tools until a handoff condition is met.

        :param messages: List of chat messages to process
        :param kwargs: Additional keyword arguments matching the defined input types
        :return: Dictionary containing messages and outputs matching the defined output types
        """
        context = ToolContext(
            input_schema=self.input_variables,
            output_schema=self.output_variables,
            inputs=kwargs
        )

        if self.system_prompt is not None:
            messages = [ChatMessage.from_system(self.system_prompt)] + messages

        self.pipeline.warm_up()

        result = self.pipeline.run(
            data={
                "joiner": {"value": messages},
                "context_joiner": {"value": context},
            },
            include_outputs_from={"context_joiner"},
        )

        return {
            "messages": result["router"]["handoff"],
            **result["context_joiner"]["value"].outputs,
        }
