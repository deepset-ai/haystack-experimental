# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Union

from haystack import logging
from haystack.components.agents.agent import Agent as HaystackAgent
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline import AsyncPipeline, Pipeline
from haystack.core.pipeline.breakpoint import (
    _create_pipeline_snapshot_from_chat_generator,
    _create_pipeline_snapshot_from_tool_invoker,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.tools import ToolsType

from haystack_experimental.memory_stores.mem0.memory_store import Mem0MemoryStore

logger = logging.getLogger(__name__)


class Agent(HaystackAgent):
    """
    A Haystack component that implements a memory-based agent.
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        tools: Optional[ToolsType] = None,
        memory_store: Optional[Mem0MemoryStore] = None,
        system_prompt: Optional[str] = None,
        exit_conditions: Optional[list[str]] = None,
        state_schema: Optional[dict[str, Any]] = None,
        max_agent_steps: int = 100,
        streaming_callback: Optional[StreamingCallbackT] = None,
        raise_on_tool_invocation_failure: bool = False,
        tool_invoker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: List of Tool objects or a Toolset that the agent can use.
        :param memory_store: The memory store to use for the agent.
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
        super(Agent, self).__init__(
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
        self.memory_store = memory_store

    def run(  # noqa: PLR0915
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        generation_kwargs: Optional[dict[str, Any]] = None,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process messages and execute tools until an exit condition is met.

        :param messages: List of Haystack ChatMessage objects to process.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
            The same callback can be configured to emit tool results when a tool is called.
        :param generation_kwargs: Additional keyword arguments for LLM. These parameters will
            override the parameters passed during component initialization.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :param snapshot: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
            the relevant information to restart the Agent execution from where it left off.
        :param system_prompt: System prompt for the agent. If provided, it overrides the default system prompt.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the agent's run.
            - "last_message": The last message exchanged during the agent's run.
            - Any additional keys defined in the `state_schema`.
        :raises RuntimeError: If the Agent component wasn't warmed up before calling `run()`.
        :raises BreakpointException: If an agent breakpoint is triggered.
        """
        # We pop parent_snapshot from kwargs to avoid passing it into State.
        parent_snapshot = kwargs.pop("parent_snapshot", None)
        retrieved_memory: list[ChatMessage] = []
        updated_system_prompt = system_prompt

        # Retrieve memories from the memory store
        if self.memory_store:
            retrieved_memory = self.memory_store.search_memories(query=messages[-1].text)

        if retrieved_memory:
            memory_instruction = (
                "\n\nWhen messages start with `[MEMORY]`, treat them as long-term "
                "context and use them to guide the response if relevant."
            )
            updated_system_prompt = f"{system_prompt}{memory_instruction}"

            memory_text = "\n".join(
                f"- MEMORY #{idx + 1}: {memory.text}" for idx, memory in enumerate(retrieved_memory)
            )

            memory_message = ChatMessage.from_system(
                text=f"Here are the relevant memories for the user's query: {memory_text}",
                meta=retrieved_memory[0].meta,
            )
            memory_messages = [memory_message]
        else:
            memory_messages = []

        combined_messages = messages + memory_messages
        agent_inputs = {
            "messages": combined_messages,
            "streaming_callback": streaming_callback,
            "break_point": break_point,
            "snapshot": snapshot,
            **kwargs,
        }
        self._runtime_checks(break_point=break_point)

        if snapshot:
            exe_context = self._initialize_from_snapshot(
                snapshot=snapshot,
                streaming_callback=streaming_callback,
                requires_async=False,
                tools=tools,
                generation_kwargs=generation_kwargs,
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=combined_messages,
                streaming_callback=streaming_callback,
                requires_async=False,
                system_prompt=updated_system_prompt,
                tools=tools,
                generation_kwargs=generation_kwargs,
                **kwargs,
            )

        with self._create_agent_span() as span:
            span.set_content_tag("haystack.agent.input", _deepcopy_with_exceptions(agent_inputs))

            while exe_context.counter < self.max_agent_steps:
                # Handle breakpoint and ChatGenerator call
                Agent._check_chat_generator_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                # We skip the chat generator when restarting from a snapshot from a ToolBreakpoint
                if exe_context.skip_chat_generator:
                    llm_messages = exe_context.state.get("messages", [])[-1:]
                    # Set to False so the next iteration will call the chat generator
                    exe_context.skip_chat_generator = False
                else:
                    try:
                        result = Pipeline._run_component(
                            component_name="chat_generator",
                            component={"instance": self.chat_generator},
                            inputs={
                                "messages": exe_context.state.data["messages"],
                                **exe_context.chat_generator_inputs,
                            },
                            component_visits=exe_context.component_visits,
                            parent_span=span,
                        )
                    except PipelineRuntimeError as e:
                        pipeline_snapshot = _create_pipeline_snapshot_from_chat_generator(
                            agent_name=getattr(self, "__component_name__", None),
                            execution_context=exe_context,
                            parent_snapshot=parent_snapshot,
                        )
                        e.pipeline_snapshot = pipeline_snapshot
                        raise e

                    llm_messages = result["replies"]
                    exe_context.state.set("messages", llm_messages)

                # Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    exe_context.counter += 1
                    break

                # Handle breakpoint and ToolInvoker call
                Agent._check_tool_invoker_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                try:
                    # We only send the messages from the LLM to the tool invoker
                    tool_invoker_result = Pipeline._run_component(
                        component_name="tool_invoker",
                        component={"instance": self._tool_invoker},
                        inputs={
                            "messages": llm_messages,
                            "state": exe_context.state,
                            **exe_context.tool_invoker_inputs,
                        },
                        component_visits=exe_context.component_visits,
                        parent_span=span,
                    )
                except PipelineRuntimeError as e:
                    # Access the original Tool Invoker exception
                    original_error = e.__cause__
                    tool_name = getattr(original_error, "tool_name", None)

                    pipeline_snapshot = _create_pipeline_snapshot_from_tool_invoker(
                        tool_name=tool_name,
                        agent_name=getattr(self, "__component_name__", None),
                        execution_context=exe_context,
                        parent_snapshot=parent_snapshot,
                    )
                    e.pipeline_snapshot = pipeline_snapshot
                    raise e

                tool_messages = tool_invoker_result["tool_messages"]
                exe_context.state = tool_invoker_result["state"]
                exe_context.state.set("messages", tool_messages)

                # Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    exe_context.counter += 1
                    break

                # Increment the step counter
                exe_context.counter += 1

            if exe_context.counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", exe_context.state.data)
            span.set_tag("haystack.agent.steps_taken", exe_context.counter)

        result = {**exe_context.state.data}
        if msgs := result.get("messages"):
            result["messages"] = msgs
            result["last_message"] = msgs[-1] if msgs else None

            # Add the new conversation as memories to the memory store
            new_memories = [
                message for message in msgs if message.role.value == "user" or message.role.value == "assistant"
            ]
            if self.memory_store:
                self.memory_store.add_memories(new_memories)
        return result

    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        generation_kwargs: Optional[dict[str, Any]] = None,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        memory_store_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously process messages and execute tools until the exit condition is met.

        This is the asynchronous version of the `run` method. It follows the same logic but uses
        asynchronous operations where possible, such as calling the `run_async` method of the ChatGenerator
        if available.

        :param messages: List of Haystack ChatMessage objects to process.
        :param streaming_callback: An asynchronous callback that will be invoked when a response is streamed from the
            LLM. The same callback can be configured to emit tool results when a tool is called.
        :param generation_kwargs: Additional keyword arguments for LLM. These parameters will
            override the parameters passed during component initialization.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :param snapshot: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
            the relevant information to restart the Agent execution from where it left off.
        :param system_prompt: System prompt for the agent. If provided, it overrides the default system prompt.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the agent's run.
            - "last_message": The last message exchanged during the agent's run.
            - Any additional keys defined in the `state_schema`.
        :raises RuntimeError: If the Agent component wasn't warmed up before calling `run_async()`.
        :raises BreakpointException: If an agent breakpoint is triggered.
        """
        # We pop parent_snapshot from kwargs to avoid passing it into State.
        parent_snapshot = kwargs.pop("parent_snapshot", None)

        retrieved_memory: list[ChatMessage] = []
        updated_system_prompt = system_prompt

        # Retrieve memories from the memory store
        if self.memory_store:
            retrieved_memory = self.memory_store.search_memories(query=messages[-1].text)

        if retrieved_memory:
            memory_instruction = (
                "\n\nWhen messages start with `[MEMORY]`, treat them as long-term "
                "context and use them to guide the response if relevant."
            )
            updated_system_prompt = f"{system_prompt}{memory_instruction}"

            memory_messages = [
                ChatMessage.from_system(
                    text=f"[MEMORY #{idx + 1}] {memory.text}",
                    meta=memory.meta,
                )
                for idx, memory in enumerate(retrieved_memory)
            ]
        else:
            memory_messages = []

        combined_messages = messages + memory_messages
        agent_inputs = {
            "messages": combined_messages,
            "streaming_callback": streaming_callback,
            "break_point": break_point,
            "snapshot": snapshot,
            **kwargs,
        }
        self._runtime_checks(break_point=break_point)

        if snapshot:
            exe_context = self._initialize_from_snapshot(
                snapshot=snapshot,
                streaming_callback=streaming_callback,
                requires_async=True,
                tools=tools,
                generation_kwargs=generation_kwargs,
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=combined_messages,
                streaming_callback=streaming_callback,
                requires_async=True,
                system_prompt=updated_system_prompt,
                tools=tools,
                generation_kwargs=generation_kwargs,
                **kwargs,
            )

        with self._create_agent_span() as span:
            span.set_content_tag("haystack.agent.input", _deepcopy_with_exceptions(agent_inputs))

            while exe_context.counter < self.max_agent_steps:
                # Handle breakpoint and ChatGenerator call
                self._check_chat_generator_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                # We skip the chat generator when restarting from a snapshot from a ToolBreakpoint
                if exe_context.skip_chat_generator:
                    llm_messages = exe_context.state.get("messages", [])[-1:]
                    # Set to False so the next iteration will call the chat generator
                    exe_context.skip_chat_generator = False
                else:
                    result = await AsyncPipeline._run_component_async(
                        component_name="chat_generator",
                        component={"instance": self.chat_generator},
                        component_inputs={
                            "messages": exe_context.state.data["messages"],
                            **exe_context.chat_generator_inputs,
                        },
                        component_visits=exe_context.component_visits,
                        parent_span=span,
                    )
                    llm_messages = result["replies"]
                    exe_context.state.set("messages", llm_messages)

                # Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    exe_context.counter += 1
                    break

                # Handle breakpoint and ToolInvoker call
                self._check_tool_invoker_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                # We only send the messages from the LLM to the tool invoker
                tool_invoker_result = await AsyncPipeline._run_component_async(
                    component_name="tool_invoker",
                    component={"instance": self._tool_invoker},
                    component_inputs={
                        "messages": llm_messages,
                        "state": exe_context.state,
                        **exe_context.tool_invoker_inputs,
                    },
                    component_visits=exe_context.component_visits,
                    parent_span=span,
                )
                tool_messages = tool_invoker_result["tool_messages"]
                exe_context.state = tool_invoker_result["state"]
                exe_context.state.set("messages", tool_messages)

                # Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    exe_context.counter += 1
                    break

                # Increment the step counter
                exe_context.counter += 1

            if exe_context.counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", exe_context.state.data)
            span.set_tag("haystack.agent.steps_taken", exe_context.counter)

        result = {**exe_context.state.data}
        if msgs := result.get("messages"):
            result["messages"] = msgs
            result["last_message"] = msgs[-1] if msgs else None

            # Add the new conversation as memories to the memory store
            new_memories = [
                message for message in msgs if message.role.value == "user" or message.role.value == "assistant"
            ]
            if self.memory_store:
                self.memory_store.add_memories(new_memories)
        return result
