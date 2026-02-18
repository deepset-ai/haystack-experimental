import inspect
from typing import Any

from haystack import component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.human_in_the_loop.strategies import ConfirmationStrategy
from haystack.tools import ToolsType

from haystack_experimental.chat_message_stores.types import ChatMessageStore
from haystack_experimental.components.agents import Agent
from haystack_experimental.memory_stores.types import MemoryStore


@component
class UnifiedAgent(Agent):
    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        user_prompt: str,
        system_prompt: str | None = None,
        tools: ToolsType | None = None,
        exit_conditions: list[str] | None = None,
        state_schema: dict[str, Any] | None = None,
        max_agent_steps: int = 100,
        streaming_callback: StreamingCallbackT | None = None,
        raise_on_tool_invocation_failure: bool = False,
        confirmation_strategies: dict[str, ConfirmationStrategy] | None = None,
        tool_invoker_kwargs: dict[str, Any] | None = None,
        chat_message_store: ChatMessageStore | None = None,
        memory_store: MemoryStore | None = None,
    ):
        super(UnifiedAgent, self).__init__(
            chat_generator=chat_generator,
            tools=tools,
            system_prompt=system_prompt,
            exit_conditions=exit_conditions,
            state_schema=state_schema,
            max_agent_steps=max_agent_steps,
            streaming_callback=streaming_callback,
            raise_on_tool_invocation_failure=raise_on_tool_invocation_failure,
            confirmation_strategies=confirmation_strategies,
            tool_invoker_kwargs=tool_invoker_kwargs,
            chat_message_store=chat_message_store,
            memory_store=memory_store,
        )
        self.user_prompt: str = user_prompt
        self._user_prompt_builder: ChatPromptBuilder = ChatPromptBuilder(template=user_prompt)

        # Register template variables as dynamic input sockets so that pipelines can connect other components' outputs.
        # We skip variables whose names collide with existing explicit input sockets defined by the run() signature.
        run_params = set(inspect.signature(self.run).parameters.keys())
        for var in self._user_prompt_builder.variables:
            if var not in run_params:
                component.set_input_type(self, var, Any, "")

    def run(  # type: ignore[override]  # noqa: PLR0915 PLR0912 D102
        self,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        break_point: AgentBreakpoint | None = None,
        snapshot: AgentSnapshot | None = None,
        tools: ToolsType | list[str] | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
        chat_message_store_kwargs: dict[str, Any] | None = None,
        memory_store_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Distinguish between prompt builder kwargs and state kwargs.
        # Template variables go to the prompt builder; state schema keys go to the parent Agent.run().
        # A kwarg present in both sets is passed to both. Unknown kwargs (in neither set) are passed
        # only to the prompt builder, where Jinja2 silently ignores them, rather than to State which
        # would reject unknown keys.
        template_variable_names = set(self._user_prompt_builder.variables)
        state_schema_keys = set(self.state_schema.keys())

        prompt_kwargs = {k: v for k, v in kwargs.items() if k in template_variable_names or k not in state_schema_keys}
        agent_state_kwargs = {k: v for k, v in kwargs.items() if k in state_schema_keys}

        prompt_builder_result = self._user_prompt_builder.run(
            template=user_prompt or self.user_prompt,
            **prompt_kwargs,
        )
        messages: list[ChatMessage] = prompt_builder_result["prompt"]

        return super(UnifiedAgent, self).run(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            break_point=break_point,
            snapshot=snapshot,
            system_prompt=system_prompt or self.system_prompt,
            tools=tools,
            confirmation_strategy_context=confirmation_strategy_context,
            chat_message_store_kwargs=chat_message_store_kwargs,
            memory_store_kwargs=memory_store_kwargs,
            **agent_state_kwargs,
        )

    async def run_async(  # type: ignore[override] # noqa: D102
        self,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        break_point: AgentBreakpoint | None = None,
        snapshot: AgentSnapshot | None = None,
        tools: ToolsType | list[str] | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
        chat_message_store_kwargs: dict[str, Any] | None = None,
        memory_store_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # Distinguish between prompt builder kwargs and state kwargs.
        # Template variables go to the prompt builder; state schema keys go to the parent Agent.run().
        # A kwarg present in both sets is passed to both. Unknown kwargs (in neither set) are passed
        # only to the prompt builder, where Jinja2 silently ignores them, rather than to State which
        # would reject unknown keys.
        template_variable_names = set(self._user_prompt_builder.variables)
        state_schema_keys = set(self.state_schema.keys())

        prompt_kwargs = {k: v for k, v in kwargs.items() if k in template_variable_names or k not in state_schema_keys}
        agent_state_kwargs = {k: v for k, v in kwargs.items() if k in state_schema_keys}

        prompt_builder_result = self._user_prompt_builder.run(
            template=user_prompt or self.user_prompt,
            **prompt_kwargs,
        )
        messages: list[ChatMessage] = prompt_builder_result["prompt"]

        return await super(UnifiedAgent, self).run_async(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            break_point=break_point,
            snapshot=snapshot,
            system_prompt=system_prompt or self.system_prompt,
            tools=tools,
            confirmation_strategy_context=confirmation_strategy_context,
            chat_message_store_kwargs=chat_message_store_kwargs,
            memory_store_kwargs=memory_store_kwargs,
            **agent_state_kwargs,
        )
