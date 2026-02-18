# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING, Any, Literal

from haystack import Pipeline, logging, super_component
from haystack.components.agents.agent import Agent as HaystackAgent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.tools import ToolsType

logger = logging.getLogger(__name__)


@super_component
class IntegratedAgent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    NOTE: This class extends Haystack's Agent component to embed chat-prompt-building in its API.

    The component processes messages and executes tools until an exit condition is met.
    The exit condition can be triggered either by a direct text response or by invoking a specific designated tool.
    Multiple exit conditions can be specified.

    When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

    ### Usage example

    #### With a list of ChatMessage objects as template

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    from haystack_experimental.super_components.agents import IntegratedAgent

    agent = IntegratedAgent(
        chat_generator=OpenAIChatGenerator(),
        template=[ChatMessage.from_user("Tell me about {{topic}}")],
    )

    result = agent.run(topic="Haystack")
    assert "messages" in result
    assert "last_message" in result
    ```

    #### With a string template

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack_experimental.super_components.agents import IntegratedAgent

    template = \"\"\"
    {% message role="system" %}
    You are a helpful assistant that speaks {{language}}.
    {% endmessage %}

    {% message role="user" %}
    {{question}}
    {% endmessage %}
    \"\"\"

    agent = IntegratedAgent(
        chat_generator=OpenAIChatGenerator(),
        template=template,
    )

    result = agent.run(language="English", question="What is Haystack?")
    ```

    #### With tools

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool

    from haystack_experimental.super_components.agents import IntegratedAgent

    calculator_tool = Tool(
        name="calculator",
        description="A tool for performing mathematical calculations.",
        ...
    )

    agent = IntegratedAgent(
        chat_generator=OpenAIChatGenerator(),
        template=[ChatMessage.from_user("{{query}}")],
        tools=[calculator_tool],
    )

    result = agent.run(query="What is 2 + 2?")
    ```
    """

    def __init__(
        self,
        *,
        # --- ChatPromptBuilder parameters --- #
        template: list[ChatMessage] | str | None = None,
        required_variables: list[str] | Literal["*"] | None = None,
        variables: list[str] | None = None,
        # --- Agent parameters --- #
        chat_generator: ChatGenerator,
        tools: ToolsType | None = None,
        system_prompt: str | None = None,
        exit_conditions: list[str] | None = None,
        state_schema: dict[str, Any] | None = None,
        max_agent_steps: int = 100,
        streaming_callback: StreamingCallbackT | None = None,
        raise_on_tool_invocation_failure: bool = False,
        tool_invoker_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the prompt-builder, agent and connect the two.

        **ChatPromptBuilder parameters**:
        :param template:
            A list of `ChatMessage` objects or a string template. The component looks for Jinja2 template syntax and
            renders the prompt with the provided variables. Provide the template in either
            the `init` method` or the `run` method.
        :param required_variables:
            List variables that must be provided as input to ChatPromptBuilder.
            If a variable listed as required is not provided, an exception is raised.
            If set to "*", all variables found in the prompt are required. Optional.
        :param variables:
            List input variables to use in prompt templates instead of the ones inferred from the
            `template` parameter. For example, to use more variables during prompt engineering than the ones present
            in the default template, you can provide them here.

        **Agent parameters**:
        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset that the agent can use.
        :param system_prompt: System prompt for the agent.
        :param exit_conditions: List of conditions that will cause the agent to return.
            Can include "text" if the agent should return when it generates a message without tool calls,
            or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
        :param state_schema: The schema for the runtime state used by the tools.
        :param max_agent_steps: Maximum number of steps the agent will run before stopping. Defaults to 100.
            If the agent exceeds this number of steps, it will stop and return the current state.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
            The same callback can be configured to emit tool results when a tool is called.
        :param raise_on_tool_invocation_failure: Should the agent raise an exception when a tool invocation fails?
            If set to False, the exception will be turned into a chat message and passed to the LLM.
        :param tool_invoker_kwargs: Additional keyword arguments to pass to the ToolInvoker.
        :raises TypeError: If the chat_generator does not support tools parameter in its run method.
        :raises ValueError: If the exit_conditions are not valid.
        """

        prompt_builder = ChatPromptBuilder(
            template=template, required_variables=required_variables, variables=variables
        )
        agent = HaystackAgent(
            chat_generator=chat_generator,
            tools=tools,
            system_prompt=system_prompt,
            exit_conditions=exit_conditions,
            state_schema=state_schema,
            max_agent_steps=max_agent_steps,
            streaming_callback=streaming_callback,
            raise_on_tool_invocation_failure=raise_on_tool_invocation_failure,
            tool_invoker_kwargs=tool_invoker_kwargs,
        )

        pp = Pipeline()
        pp.add_component("prompt_builder", prompt_builder)
        pp.add_component("agent", agent)
        pp.connect("prompt_builder.prompt", "agent.messages")

        self.pipeline = pp
        # self.input_mapping = let the default auto-mapping handle input wiring to deal with template values.
        self.output_mapping = {"agent.messages": "messages", "agent.last_message": "last_message"}

    if TYPE_CHECKING:
        # fake method, never executed, but static analyzers will not complain about missing method
        def run(  # noqa: D102
            self,
            *,
            # --- ChatPromptBuilder parameters --- #
            template: list[ChatMessage] | str | None,
            template_variables: dict[str, Any] | None,
            # --- Agent parameters --- #
            streaming_callback: StreamingCallbackT | None,
            # --- Pipeline kwargs (+ template) --- #
            **kwargs: Any,
        ) -> dict[str, Any]: ...

        def warm_up(self) -> None:  # noqa: D102
            ...
