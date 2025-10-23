# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=wrong-import-order,wrong-import-position,ungrouped-imports
# ruff: noqa: I001

from dataclasses import dataclass
from typing import Any, Optional, Union

# Monkey patch Haystack's AgentSnapshot with our extended version
import haystack.dataclasses.breakpoints as hdb
from haystack_experimental.dataclasses.breakpoints import AgentSnapshot

hdb.AgentSnapshot = AgentSnapshot  # type: ignore[misc]

# Monkey patch Haystack's breakpoint functions with our extended versions
import haystack.core.pipeline.breakpoint as hs_breakpoint
import haystack_experimental.core.pipeline.breakpoint as exp_breakpoint

hs_breakpoint._create_agent_snapshot = exp_breakpoint._create_agent_snapshot
hs_breakpoint._create_pipeline_snapshot_from_tool_invoker = exp_breakpoint._create_pipeline_snapshot_from_tool_invoker  # type: ignore[assignment]

from haystack import logging
from haystack.components.agents.agent import Agent as HaystackAgent
from haystack.components.agents.agent import _ExecutionContext as Haystack_ExecutionContext
from haystack.components.agents.agent import _schema_from_dict
from haystack.components.agents.state import replace_values
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline import AsyncPipeline, Pipeline
from haystack.core.pipeline.breakpoint import (
    _create_pipeline_snapshot_from_chat_generator,
    _create_pipeline_snapshot_from_tool_invoker,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.core.serialization import default_from_dict, import_class_by_name
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import AgentBreakpoint, ToolBreakpoint
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.tools import ToolsType, deserialize_tools_or_toolset_inplace
from haystack.utils.callable_serialization import deserialize_callable
from haystack.utils.deserialization import deserialize_chatgenerator_inplace

from haystack_experimental.components.agents.human_in_the_loop import (
    ConfirmationStrategy,
    ToolExecutionDecision,
    HITLBreakpointException,
)
from haystack_experimental.components.agents.human_in_the_loop.strategies import _process_confirmation_strategies

logger = logging.getLogger(__name__)


@dataclass
class _ExecutionContext(Haystack_ExecutionContext):
    """
    Execution context for the Agent component

    Extends Haystack's _ExecutionContext to include tool execution decisions for human-in-the-loop strategies.

    :param tool_execution_decisions: Optional list of ToolExecutionDecision objects to use instead of prompting
        the user. This is useful when restarting from a snapshot where tool execution decisions were already made.
    """

    tool_execution_decisions: Optional[list[ToolExecutionDecision]] = None


class Agent(HaystackAgent):
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    NOTE: This class extends Haystack's Agent component to add support for human-in-the-loop confirmation strategies.

    The component processes messages and executes tools until an exit condition is met.
    The exit condition can be triggered either by a direct text response or by invoking a specific designated tool.
    Multiple exit conditions can be specified.

    When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

    ### Usage example
    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools.tool import Tool

    from haystack_experimental.components.agents import Agent
    from haystack_experimental.components.agents.human_in_the_loop import (
        HumanInTheLoopStrategy,
        AlwaysAskPolicy,
        NeverAskPolicy,
        SimpleConsoleUI,
    )

    calculator_tool = Tool(name="calculator", description="A tool for performing mathematical calculations.", ...)
    search_tool = Tool(name="search", description="A tool for searching the web.", ...)

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[calculator_tool, search_tool],
        confirmation_strategies={
            calculator_tool.name: HumanInTheLoopStrategy(
                confirmation_policy=NeverAskPolicy(), confirmation_ui=SimpleConsoleUI()
            ),
            search_tool.name: HumanInTheLoopStrategy(
                confirmation_policy=AlwaysAskPolicy(), confirmation_ui=SimpleConsoleUI()
            ),
        },
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
        chat_generator: ChatGenerator,
        tools: Optional[ToolsType] = None,
        system_prompt: Optional[str] = None,
        exit_conditions: Optional[list[str]] = None,
        state_schema: Optional[dict[str, Any]] = None,
        max_agent_steps: int = 100,
        streaming_callback: Optional[StreamingCallbackT] = None,
        raise_on_tool_invocation_failure: bool = False,
        confirmation_strategies: Optional[dict[str, ConfirmationStrategy]] = None,
        tool_invoker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: List of Tool objects or a Toolset that the agent can use.
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
        self._confirmation_strategies = confirmation_strategies or {}

    def _initialize_fresh_execution(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT],
        requires_async: bool,
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        **kwargs: dict[str, Any],
    ) -> _ExecutionContext:
        """
        Initialize execution context for a fresh run of the agent.

        :param messages: List of ChatMessage objects to start the agent with.
        :param streaming_callback: Optional callback for streaming responses.
        :param requires_async: Whether the agent run requires asynchronous execution.
        :param system_prompt: System prompt for the agent. If provided, it overrides the default system prompt.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :param kwargs: Additional data to pass to the State used by the Agent.
        """
        exe_context = super(Agent, self)._initialize_fresh_execution(
            messages=messages,
            streaming_callback=streaming_callback,
            requires_async=requires_async,
            system_prompt=system_prompt,
            tools=tools,
            **kwargs,
        )
        # NOTE: 1st difference with parent method to add this to tool_invoker_inputs
        if self._tool_invoker:
            exe_context.tool_invoker_inputs["enable_streaming_callback_passthrough"] = (
                self._tool_invoker.enable_streaming_callback_passthrough
            )
        # NOTE: 2nd difference is to use the extended _ExecutionContext
        return _ExecutionContext(
            state=exe_context.state,
            component_visits=exe_context.component_visits,
            chat_generator_inputs=exe_context.chat_generator_inputs,
            tool_invoker_inputs=exe_context.tool_invoker_inputs,
        )

    def _initialize_from_snapshot(  # type: ignore[override]
        self,
        snapshot: AgentSnapshot,
        streaming_callback: Optional[StreamingCallbackT],
        requires_async: bool,
        *,
        tools: Optional[Union[ToolsType, list[str]]] = None,
    ) -> _ExecutionContext:
        """
        Initialize execution context from an AgentSnapshot.

        :param snapshot: An AgentSnapshot containing the state of a previously saved agent execution.
        :param streaming_callback: Optional callback for streaming responses.
        :param requires_async: Whether the agent run requires asynchronous execution.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        """
        exe_context = super(Agent, self)._initialize_from_snapshot(
            snapshot=snapshot, streaming_callback=streaming_callback, requires_async=requires_async, tools=tools
        )
        # NOTE: 1st difference with parent method to add this to tool_invoker_inputs
        if self._tool_invoker:
            exe_context.tool_invoker_inputs["enable_streaming_callback_passthrough"] = (
                self._tool_invoker.enable_streaming_callback_passthrough
            )
        # NOTE: 2nd difference is to use the extended _ExecutionContext and add tool_execution_decisions
        return _ExecutionContext(
            state=exe_context.state,
            component_visits=exe_context.component_visits,
            chat_generator_inputs=exe_context.chat_generator_inputs,
            tool_invoker_inputs=exe_context.tool_invoker_inputs,
            counter=exe_context.counter,
            skip_chat_generator=exe_context.skip_chat_generator,
            tool_execution_decisions=snapshot.tool_execution_decisions,
        )

    def run(  # noqa: PLR0915
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,  # type: ignore[override]
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process messages and execute tools until an exit condition is met.

        :param messages: List of Haystack ChatMessage objects to process.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
            The same callback can be configured to emit tool results when a tool is called.
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
        agent_inputs = {
            "messages": messages,
            "streaming_callback": streaming_callback,
            "break_point": break_point,
            "snapshot": snapshot,
            **kwargs,
        }
        self._runtime_checks(break_point=break_point, snapshot=snapshot)

        if snapshot:
            exe_context = self._initialize_from_snapshot(
                snapshot=snapshot, streaming_callback=streaming_callback, requires_async=False, tools=tools
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=messages,
                streaming_callback=streaming_callback,
                requires_async=False,
                system_prompt=system_prompt,
                tools=tools,
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

                # Apply confirmation strategies and update State and messages sent to ToolInvoker
                try:
                    # Run confirmation strategies to get updated tool call messages and modified chat history
                    modified_tool_call_messages, new_chat_history = _process_confirmation_strategies(
                        confirmation_strategies=self._confirmation_strategies,
                        messages_with_tool_calls=llm_messages,
                        execution_context=exe_context,
                    )
                    # Replace the chat history in state with the modified one
                    exe_context.state.set(key="messages", value=new_chat_history, handler_override=replace_values)
                except HITLBreakpointException as tbp_error:
                    # We create a break_point to pass into _check_tool_invoker_breakpoint
                    break_point = AgentBreakpoint(
                        agent_name=getattr(self, "__component_name__", ""),
                        break_point=ToolBreakpoint(
                            component_name="tool_invoker",
                            tool_name=tbp_error.tool_name,
                            visit_count=exe_context.component_visits["tool_invoker"],
                            snapshot_file_path=tbp_error.snapshot_file_path,
                        ),
                    )

                # Handle breakpoint
                Agent._check_tool_invoker_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )

                # Run ToolInvoker
                try:
                    # We only send the messages from the LLM to the tool invoker
                    tool_invoker_result = Pipeline._run_component(
                        component_name="tool_invoker",
                        component={"instance": self._tool_invoker},
                        inputs={
                            "messages": modified_tool_call_messages,
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

                # Set execution context tool execution decisions to empty after applying them b/c they should only
                # be used once for the current tool calls
                exe_context.tool_execution_decisions = None
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
            result["last_message"] = msgs[-1]
        return result

    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,  # type: ignore[override]
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
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
        agent_inputs = {
            "messages": messages,
            "streaming_callback": streaming_callback,
            "break_point": break_point,
            "snapshot": snapshot,
            **kwargs,
        }
        self._runtime_checks(break_point=break_point, snapshot=snapshot)

        if snapshot:
            exe_context = self._initialize_from_snapshot(
                snapshot=snapshot, streaming_callback=streaming_callback, requires_async=True, tools=tools
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=messages,
                streaming_callback=streaming_callback,
                requires_async=True,
                system_prompt=system_prompt,
                tools=tools,
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

                # Apply confirmation strategies and update State and messages sent to ToolInvoker
                try:
                    # Run confirmation strategies to get updated tool call messages and modified chat history
                    modified_tool_call_messages, new_chat_history = _process_confirmation_strategies(
                        confirmation_strategies=self._confirmation_strategies,
                        messages_with_tool_calls=llm_messages,
                        execution_context=exe_context,
                    )
                    # Replace the chat history in state with the modified one
                    exe_context.state.set(key="messages", value=new_chat_history, handler_override=replace_values)
                except HITLBreakpointException as tbp_error:
                    # We create a break_point to pass into _check_tool_invoker_breakpoint
                    break_point = AgentBreakpoint(
                        agent_name=getattr(self, "__component_name__", ""),
                        break_point=ToolBreakpoint(
                            component_name="tool_invoker",
                            tool_name=tbp_error.tool_name,
                            visit_count=exe_context.component_visits["tool_invoker"],
                            snapshot_file_path=tbp_error.snapshot_file_path,
                        ),
                    )

                # Handle breakpoint
                Agent._check_tool_invoker_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )

                # Run ToolInvoker
                # We only send the messages from the LLM to the tool invoker
                tool_invoker_result = await AsyncPipeline._run_component_async(
                    component_name="tool_invoker",
                    component={"instance": self._tool_invoker},
                    component_inputs={
                        "messages": modified_tool_call_messages,
                        "state": exe_context.state,
                        **exe_context.tool_invoker_inputs,
                    },
                    component_visits=exe_context.component_visits,
                    parent_span=span,
                )

                # Set execution context tool execution decisions to empty after applying them b/c they should only
                # be used once for the current tool calls
                exe_context.tool_execution_decisions = None
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
            result["last_message"] = msgs[-1]
        return result

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        data = super(Agent, self).to_dict()
        data["init_parameters"]["confirmation_strategies"] = (
            {name: strategy.to_dict() for name, strategy in self._confirmation_strategies.items()}
            if self._confirmation_strategies
            else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        """
        Deserialize the agent from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized agent
        """
        init_params = data.get("init_parameters", {})

        deserialize_chatgenerator_inplace(init_params, key="chat_generator")

        if "state_schema" in init_params:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_or_toolset_inplace(init_params, key="tools")

        if "confirmation_strategies" in init_params and init_params["confirmation_strategies"] is not None:
            for name, strategy_dict in init_params["confirmation_strategies"].items():
                strategy_class = import_class_by_name(strategy_dict["type"])
                if not hasattr(strategy_class, "from_dict"):
                    raise TypeError(f"{strategy_class} does not have from_dict method implemented.")
                init_params["confirmation_strategies"][name] = strategy_class.from_dict(strategy_dict)

        return default_from_dict(cls, data)
