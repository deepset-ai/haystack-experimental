# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import inspect
from typing import Any, Optional, Union

from haystack import logging
from haystack import component
from haystack.core.pipeline.pipeline import Pipeline
from haystack.components.agents.agent import Agent as HaystackAgent
from haystack.components.agents.agent import (
    _ExecutionContext,
    State,
    _schema_from_dict,
    _schema_to_dict,
    _validate_schema,
    merge_lists,
)
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.tools.tool_invoker import ToolInvoker
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline.breakpoint import (
    _create_pipeline_snapshot_from_chat_generator,
    _create_pipeline_snapshot_from_tool_invoker,
    _validate_tool_breakpoint_is_valid,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict, import_class_by_name
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot, ToolBreakpoint
from haystack.dataclasses.streaming_chunk import StreamingCallbackT, select_streaming_callback
from haystack.tools import Tool, Toolset, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_chatgenerator_inplace
from haystack.utils import _deserialize_value_with_schema

from haystack_experimental.components.agents.human_in_the_loop import ConfirmationStrategy
from haystack_experimental.components.agents.human_in_the_loop.errors import ToolBreakpointException

logger = logging.getLogger(__name__)


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

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        tools: Optional[Union[list[Tool], Toolset]] = None,
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
        # Check if chat_generator supports tools parameter
        chat_generator_run_method = inspect.signature(chat_generator.run)
        if "tools" not in chat_generator_run_method.parameters:
            raise TypeError(
                f"{type(chat_generator).__name__} does not accept tools parameter in its run method. "
                "The Agent component requires a chat generator that supports tools."
            )

        valid_exits = ["text"] + [tool.name for tool in tools or []]
        if exit_conditions is None:
            exit_conditions = ["text"]
        if not all(condition in valid_exits for condition in exit_conditions):
            raise ValueError(
                f"Invalid exit conditions provided: {exit_conditions}. "
                f"Valid exit conditions must be a subset of {valid_exits}. "
                "Ensure that each exit condition corresponds to either 'text' or a valid tool name."
            )

        # Validate state schema if provided
        if state_schema is not None:
            _validate_schema(state_schema)
        self._state_schema = state_schema or {}

        # Initialize state schema
        resolved_state_schema = _deepcopy_with_exceptions(self._state_schema)
        if resolved_state_schema.get("messages") is None:
            resolved_state_schema["messages"] = {"type": list[ChatMessage], "handler": merge_lists}
        self.state_schema = resolved_state_schema

        self.chat_generator = chat_generator
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_conditions = exit_conditions
        self.max_agent_steps = max_agent_steps
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback

        output_types = {"last_message": ChatMessage}
        for param, config in self.state_schema.items():
            output_types[param] = config["type"]
            # Skip setting input types for parameters that are already in the run method
            if param in ["messages", "streaming_callback"]:
                continue
            component.set_input_type(self, name=param, type=config["type"], default=None)
        component.set_output_types(self, **output_types)

        self._confirmation_strategies = confirmation_strategies or {}
        self.tool_invoker_kwargs = tool_invoker_kwargs
        self._tool_invoker = None
        if self.tools:
            resolved_tool_invoker_kwargs = {
                "tools": self.tools,
                "raise_on_failure": self.raise_on_tool_invocation_failure,
                **(tool_invoker_kwargs or {}),
            }
            self._tool_invoker = ToolInvoker(**resolved_tool_invoker_kwargs)
        else:
            logger.warning(
                "No tools provided to the Agent. The Agent will behave like a ChatGenerator and only return text "
                "responses. To enable tool usage, pass tools directly to the Agent, not to the chat_generator."
            )

        self._is_warmed_up = False

    def _initialize_fresh_execution(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT],
        requires_async: bool,
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[list[Tool], Toolset, list[str]]] = None,
        **kwargs,
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
        system_prompt = system_prompt or self.system_prompt
        if system_prompt is not None:
            messages = [ChatMessage.from_system(system_prompt)] + messages

        if all(m.is_from(ChatRole.SYSTEM) for m in messages):
            logger.warning("All messages provided to the Agent component are system messages. This is not recommended.")

        state = State(schema=self.state_schema, data=kwargs)
        state.set("messages", messages)

        streaming_callback = select_streaming_callback(  # type: ignore[call-overload]
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=requires_async
        )

        selected_tools = self._select_tools(tools)
        tool_invoker_inputs: dict[str, Any] = {"tools": selected_tools}
        generator_inputs: dict[str, Any] = {"tools": selected_tools}
        if streaming_callback is not None:
            tool_invoker_inputs["streaming_callback"] = streaming_callback
            tool_invoker_inputs["enable_streaming_callback_passthrough"] = (
                self._tool_invoker.enable_streaming_callback_passthrough
            )
            generator_inputs["streaming_callback"] = streaming_callback

        return _ExecutionContext(
            state=state,
            component_visits=dict.fromkeys(["chat_generator", "tool_invoker"], 0),
            chat_generator_inputs=generator_inputs,
            tool_invoker_inputs=tool_invoker_inputs,
        )

    def _initialize_from_snapshot(
        self,
        snapshot: AgentSnapshot,
        streaming_callback: Optional[StreamingCallbackT],
        requires_async: bool,
        *,
        tools: Optional[Union[list[Tool], Toolset, list[str]]] = None,
    ) -> _ExecutionContext:
        """
        Initialize execution context from an AgentSnapshot.

        :param snapshot: An AgentSnapshot containing the state of a previously saved agent execution.
        :param streaming_callback: Optional callback for streaming responses.
        :param requires_async: Whether the agent run requires asynchronous execution.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        """
        component_visits = snapshot.component_visits
        current_inputs = {
            "chat_generator": _deserialize_value_with_schema(snapshot.component_inputs["chat_generator"]),
            "tool_invoker": _deserialize_value_with_schema(snapshot.component_inputs["tool_invoker"]),
        }

        state_data = current_inputs["tool_invoker"]["state"].data
        state = State(schema=self.state_schema, data=state_data)

        # NOTE: Only difference from parent class is to make this check more robust
        # Handles edge case where restarting from a snapshot with an updated chat history where the last message
        # is a tool call result (e.g. after human feedback like a rejection)
        skip_chat_generator = False
        if isinstance(snapshot.break_point.break_point, ToolBreakpoint) and state.get("messages")[-1].is_from(
            "assistant"
        ):
            skip_chat_generator = True

        streaming_callback = current_inputs["chat_generator"].get("streaming_callback", streaming_callback)
        streaming_callback = select_streaming_callback(  # type: ignore[call-overload]
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=requires_async
        )

        selected_tools = self._select_tools(tools)
        tool_invoker_inputs: dict[str, Any] = {"tools": selected_tools}
        generator_inputs: dict[str, Any] = {"tools": selected_tools}
        if streaming_callback is not None:
            tool_invoker_inputs["streaming_callback"] = streaming_callback
            tool_invoker_inputs["enable_streaming_callback_passthrough"] = (
                self._tool_invoker.enable_streaming_callback_passthrough
            )
            generator_inputs["streaming_callback"] = streaming_callback

        return _ExecutionContext(
            state=state,
            component_visits=component_visits,
            chat_generator_inputs=generator_inputs,
            tool_invoker_inputs=tool_invoker_inputs,
            counter=snapshot.break_point.break_point.visit_count,
            skip_chat_generator=skip_chat_generator,
        )

    def _runtime_checks(self, break_point: Optional[AgentBreakpoint], snapshot: Optional[AgentSnapshot]) -> None:
        """
        Perform runtime checks before running the agent.

        NOTE: This differs from the parent class by allowing the agent to run with a snapshot and break point
        at the same time.

        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :param snapshot: An AgentSnapshot containing the state of a previously saved agent execution.
        :raises RuntimeError: If the Agent component wasn't warmed up before calling `run()`.
        :raises ValueError: If both break_point and snapshot are provided, or if the break_point is invalid.
        """
        if not self._is_warmed_up and hasattr(self.chat_generator, "warm_up"):
            raise RuntimeError("The component Agent wasn't warmed up. Run 'warm_up()' before calling 'run()'.")

        # Provide some warnings for potentially unintended usage if both snapshot and break_point are provided
        if break_point and snapshot:
            if break_point.break_point.visit_count < snapshot.component_visits.get(
                break_point.break_point.component_name, 0
            ):
                logger.warning(
                    "The visit_count of the provided break_point is less than the visit count in the snapshot. "
                    "This may cause the breakpoint to never trigger."
                )
            elif break_point.break_point.visit_count == snapshot.component_visits.get(
                break_point.break_point.component_name, 0
            ):
                logger.warning(
                    "The visit_count of the provided break_point is equal to the visit count in the snapshot. "
                    "This may cause the breakpoint to trigger immediately."
                )

        if break_point and isinstance(break_point.break_point, ToolBreakpoint):
            _validate_tool_breakpoint_is_valid(agent_breakpoint=break_point, tools=self.tools)

    def run(  # noqa: PLR0915
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[list[Tool], Toolset, list[str]]] = None,
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
                # Only send confirmed + modified tool calls to the ToolInvoker, but keep original messages in State
                # TODO To handle breakpoints it will be possible for this function to return a BreakpointException??
                #   - When this happens we should trigger a tool_invoker_breakpoint
                #   - On restart from snapshot we need to tell confirmation strategy the tool execution decision
                try:
                    modified_tool_call_messages, tool_call_result_messages = self._handle_confirmation_strategies(
                        messages_with_tool_calls=llm_messages,
                        execution_context=exe_context,
                    )
                except ToolBreakpointException as tbp_error:
                    # We don't raise since Agent._check_tool_invoker_breakpoint will raise the final BreakpointException
                    Agent._check_tool_invoker_breakpoint(
                        execution_context=exe_context,
                        break_point=tbp_error.break_point,
                        parent_snapshot=parent_snapshot,
                    )
                # Add Tool Call Result messages to the chat history
                exe_context.state.set("messages", tool_call_result_messages)

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

    def _handle_confirmation_strategies(
        self,
        *,
        messages_with_tool_calls: list[ChatMessage],
        execution_context: _ExecutionContext,
    ) -> tuple[list[ChatMessage], list[ChatMessage]]:
        """
        Prepare tool call parameters for execution and collect any error messages.

        :param messages_with_tool_calls: Messages containing tool calls to process
        :param execution_context: The current execution context containing state and inputs
        :returns: Tuple of modified messages with confirmed tool calls and tool call result messages
        """
        state = execution_context.state
        streaming_callback = execution_context.chat_generator_inputs.get("streaming_callback")
        tools_with_names = {tool.name: tool for tool in execution_context.tool_invoker_inputs["tools"]}
        enable_streaming_passthrough = execution_context.tool_invoker_inputs.get("enable_streaming_passthrough", False)

        modified_tool_call_messages = []
        tool_call_result_messages = []

        for message in messages_with_tool_calls:
            confirmed_tool_calls = []
            for tool_call in message.tool_calls:
                tool_name = tool_call.tool_name
                tool_to_invoke = tools_with_names[tool_name]

                # Combine user + state inputs
                llm_args = tool_call.arguments.copy()
                final_args = ToolInvoker._inject_state_args(tool_to_invoke, llm_args, state)

                # Check whether to inject streaming_callback
                if (
                    enable_streaming_passthrough
                    and streaming_callback is not None
                    and "streaming_callback" not in final_args
                    and "streaming_callback" in ToolInvoker._get_func_params(tool_to_invoke)
                ):
                    final_args["streaming_callback"] = streaming_callback

                # Handle confirmation strategies
                # If no confirmation strategy is defined for this tool, proceed with execution
                if tool_name not in self._confirmation_strategies:
                    confirmed_tool_calls.append(tool_call)
                    continue

                tool_execution_decision = self._confirmation_strategies[tool_name].run(
                    tool_name=tool_name, tool_description=tool_to_invoke.description, tool_params=final_args
                )

                if tool_execution_decision.execute:
                    # TODO Need to figure out a way to forward the feedback message here if not empty
                    #      This is relevant if tool params were modified by the user
                    # additional_feedback = tool_execution_decision.feedback
                    final_args.update(tool_execution_decision.final_tool_params)
                    confirmed_tool_calls.append(replace(tool_call, arguments=final_args))
                else:
                    # Tool execution was rejected with a message
                    tool_call_result_messages.append(
                        ChatMessage.from_tool(
                            tool_result=tool_execution_decision.feedback or "",
                            origin=tool_call,
                            error=True,
                        )
                    )

            # Update message with only confirmed tool calls
            if confirmed_tool_calls:
                modified_tool_call_messages.append(
                    ChatMessage.from_assistant(
                        text=message.text,
                        meta=message.meta,
                        name=message.name,
                        tool_calls=confirmed_tool_calls,
                        reasoning=message.reasoning,
                    )
                )

        return modified_tool_call_messages, tool_call_result_messages

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            tools=serialize_tools_or_toolset(self.tools),
            system_prompt=self.system_prompt,
            exit_conditions=self.exit_conditions,
            # We serialize the original state schema, not the resolved one to reflect the original user input
            state_schema=_schema_to_dict(self._state_schema),
            max_agent_steps=self.max_agent_steps,
            streaming_callback=serialize_callable(self.streaming_callback) if self.streaming_callback else None,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            tool_invoker_kwargs=self.tool_invoker_kwargs,
            confirmation_strategies={
                name: strategy.to_dict() for name, strategy in self._confirmation_strategies.items()
            }
            if self._confirmation_strategies
            else None,
        )

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
