from typing import Dict, Any, List, Optional, Type

from haystack import component, Pipeline, default_from_dict, default_to_dict
from haystack.components.joiners import BranchJoiner
from haystack.components.routers.conditional_router import ConditionalRouter
from haystack.dataclasses import ChatMessage
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.utils.type_serialization import serialize_type, deserialize_type

from haystack.components.tools import ToolInvoker
from haystack.tools.tool import Tool
from haystack_experimental.components.generators.anthropic.chat import AnthropicChatGenerator
from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator

_PROVIDER_GENERATOR_MAPPING = {
    "openai": OpenAIChatGenerator,
    "anthropic": AnthropicChatGenerator
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
    from haystack_experimental.dataclasses.tool import Tool

    # Define tools and input/output types
    tools = [Tool(name="calculator", description="..."), Tool(name="search", description="...")]
    input_types = {"search_depth": int}
    output_types = {"total_tokens": int}

    # Initialize the agent
    agent = Agent(
        model="anthropic:claude-3",
        generation_kwargs={"temperature": 0.7},
        tools=tools,
        handoff="search",
        context_variables=input_types,
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
            tools: List[Tool],
            system_prompt: Optional[str] = None,
            handoff: str = "text",
            generator_kwargs: Optional[Dict[str, Any]] = None,
            context_variables: Optional[Dict[str, Type]] = None,
            output_variables: Optional[Dict[str, Type]] = None,
    ):
        """
        Initialize the agent component.

        :param model: Model identifier in the format "provider:model_name"
        :param generation_kwargs: Keyword arguments for the chat model generator
        :param tools: List of Tool objects available to the agent
        :param handoff: Either "text" or name of a tool to hand off to
        :param context_variables: Dictionary mapping input variable names to their types
        :param output_variables: Dictionary mapping output variable names to their types
        :raises ValueError: If model string format is invalid or handoff is not valid
        """
        if ":" not in model:
            raise ValueError("Model string must be in format 'provider:model_name'")

        provider, model_name = model.split(":")
        if provider not in _PROVIDER_GENERATOR_MAPPING:
            raise ValueError(f"Provider must be one of {list(_PROVIDER_GENERATOR_MAPPING.keys())}")

        valid_handoffs = ["text"] + [tool.name for tool in tools]
        if handoff not in valid_handoffs:
            raise ValueError(f"Handoff must be one of {valid_handoffs}")

        # Store instance variables
        self.model = model
        self.generation_kwargs = generator_kwargs or {}
        self.tools = tools
        self.system_prompt = system_prompt
        self.handoff = handoff
        self.context_variables = context_variables or {}
        self.output_variables = output_variables or {}

        # Set input/output types
        input_types = {"messages": List[ChatMessage]}
        input_types.update(self.context_variables)
        component.set_input_types(
            instance=self,
            **input_types
        )

        output_types = {"messages": List[ChatMessage]}
        output_types.update(self.output_variables)
        component.set_output_types(
            instance=self,
            **output_types
        )

        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the component pipeline with all necessary components and connections."""
        provider, model_name = self.model.split(":")

        # Initialize components
        generator = _PROVIDER_GENERATOR_MAPPING[provider](
            model=model_name,
            tools=self.tools,
            **self.generation_kwargs
        )
        joiner = BranchJoiner(type_=List[ChatMessage])
        tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=False)

        # Create output adapter
        output_adapter = OutputAdapter(
            template="{%- set assistant_msg = (llm_messages[0].text|trim or 'Tool:')|assistant_message(llm_messages[0].tool_calls) %}{%- if tool_result %}{{ messages + [assistant_msg] + tool_result }}{%- else %}{{ messages + [assistant_msg] }}{%- endif %}",
            custom_filters={
                "create_message": ChatMessage.from_tool,
                "assistant_message": ChatMessage.from_assistant,
            },
            unsafe=True,
            output_type=List[ChatMessage]
        )

        # Configure router conditions
        if self.handoff == "text":
            handoff_condition = "{{ llm_messages[0].tool_call is none }}"
        else:
            handoff_condition = "{{ llm_messages[0].tool_call is not none and llm_messages[0].tool_call.tool_name == '" + self.handoff + "' }}"

        routes = [
            {
                "condition": handoff_condition,
                "output": "{{ original_messages + llm_messages + tool_messages }}",
                "output_type": List[ChatMessage],
                "output_name": "handoff"
            },
            {
                "condition": "{{ True }}",  # Default route
                "output": "{{ original_messages + llm_messages + tool_messages }}",
                "output_type": List[ChatMessage],
                "output_name": "continue"
            }
        ]

        router = ConditionalRouter(routes=routes, unsafe=True)

        # Set up pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_component(instance=generator, name="generator")
        self.pipeline.add_component(instance=tool_invoker, name="tool_invoker")
        self.pipeline.add_component(instance=router, name="router")
        self.pipeline.add_component(instance=joiner, name="joiner")

        # Connect components
        self.pipeline.connect("joiner.value", "generator.messages")
        self.pipeline.connect("generator.replies", "router.llm_messages")
        self.pipeline.connect("joiner.value", "router.original_messages")
        self.pipeline.connect("generator.replies", "tool_invoker.messages")
        self.pipeline.connect("tool_invoker.messages", "router.tool_messages")
        self.pipeline.connect("router.continue", "joiner.value")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        # Serialize type annotations
        serialized_context_variables = {
            name: serialize_type(type_)
            for name, type_ in self.context_variables.items()
        }
        serialized_output_variables = {
            name: serialize_type(type_)
            for name, type_ in self.output_variables.items()
        }

        return default_to_dict(
            self,
            model=self.model,
            generation_kwargs=self.generation_kwargs,
            tools=self.tools,
            system_prompt=self.system_prompt,
            handoff=self.handoff,
            context_variables=serialized_context_variables,
            output_variables=serialized_output_variables
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
        if "context_variables" in init_params:
            init_params["context_variables"] = {
                name: deserialize_type(type_)
                for name, type_ in init_params["context_variables"].items()
            }
        if "output_variables" in init_params:
            init_params["output_variables"] = {
                name: deserialize_type(type_)
                for name, type_ in init_params["output_variables"].items()
            }

        return default_from_dict(cls, data)

    def run(self, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """
        Process messages and execute tools until a handoff condition is met.

        :param messages: List of chat messages to process
        :param kwargs: Additional keyword arguments matching the defined input types
        :return: Dictionary containing messages and outputs matching the defined output types
        """
        invocation_output = {}

        if self.system_prompt is not None:
            messages = [ChatMessage.from_system(self.system_prompt)] + messages
        self.pipeline.warm_up()

        result = self.pipeline.run(
            data={
                "joiner": {"value": messages},
                "tool_invoker": {"context_variables": kwargs},
                #"output_var_joiner": {"value": invocation_output}
            }
        )

        return {"messages": result["router"]["handoff"]}