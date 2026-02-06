# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from pathlib import Path
from typing import Any, Optional

import pytest
from haystack import Pipeline, component
from haystack.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    ConfirmationUIResult,
    ToolExecutionDecision
)
from haystack.human_in_the_loop.types import ConfirmationUI
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage, ToolCall, PipelineSnapshot
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.components.agents.human_in_the_loop import BreakpointConfirmationStrategy
from haystack_experimental.components.agents.human_in_the_loop.breakpoint import (
    get_tool_calls_and_descriptions_from_snapshot,
)
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack_experimental.memory_stores.mem0 import Mem0MemoryStore


@pytest.fixture
def store():
    msg_store = InMemoryChatMessageStore()
    yield msg_store
    msg_store.delete_all_messages()

@component
class MockChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant("This is a mock response.")]}


@component
class MockChatGeneratorToolsResponse:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any) -> dict[str, list[ChatMessage]]:
        return {
            "replies": [
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name="addition_tool", arguments={"a": 2, "b": 3})]
                )
            ]
        }


@component
class MockAgent:
    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt

    @component.output_types(messages=list[ChatMessage], last_message=ChatMessage)
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        if self.system_prompt:
            system_msg = ChatMessage.from_system(self.system_prompt)
            messages = [system_msg, *messages]

        assistant_msg = ChatMessage.from_assistant("This is a mock response.")
        return {"messages": [*messages, assistant_msg], "last_message": assistant_msg}


class MockUserInterface(ConfirmationUI):
    def __init__(self, ui_result: ConfirmationUIResult) -> None:
        self.ui_result = ui_result

    def get_user_confirmation(
        self, tool_name: str, tool_description: str, tool_params: dict[str, Any]
    ) -> ConfirmationUIResult:
        return self.ui_result


def frontend_simulate_tool_decision(
    tool_calls: list[dict[str, Any]],
    tool_descriptions: dict[str, str],
    confirmation_ui_result: ConfirmationUIResult,
) -> list[dict]:
    confirmation_strategy = BlockingConfirmationStrategy(
        confirmation_policy=AlwaysAskPolicy(),
        confirmation_ui=MockUserInterface(ui_result=confirmation_ui_result),
    )

    tool_execution_decisions = []
    for tc in tool_calls:
        tool_execution_decisions.append(
            confirmation_strategy.run(
                tool_name=tc["tool_name"],
                tool_description=tool_descriptions[tc["tool_name"]],
                tool_call_id=tc["id"],
                tool_params=tc["arguments"],
            )
        )
    return [ted.to_dict() for ted in tool_execution_decisions]


def get_latest_snapshot(snapshot_file_path: str) -> PipelineSnapshot:
    snapshot_dir = Path(snapshot_file_path)
    possible_snapshots = [snapshot_dir / f for f in os.listdir(snapshot_dir)]
    latest_snapshot_file = str(max(possible_snapshots, key=os.path.getctime))
    return load_pipeline_snapshot(latest_snapshot_file)


def run_agent(
    agent: Agent,
    messages: list[ChatMessage],
    snapshot_file_path: Optional[str] = None,
    tool_execution_decisions: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    # Load the latest snapshot if a path is provided
    snapshot = None
    if snapshot_file_path:
        snapshot = get_latest_snapshot(snapshot_file_path=snapshot_file_path)

        # Add any new tool execution decisions to the snapshot
        if tool_execution_decisions:
            teds = [ToolExecutionDecision.from_dict(ted) for ted in tool_execution_decisions]
            existing_decisions = snapshot.agent_snapshot.tool_execution_decisions or []
            snapshot.agent_snapshot.tool_execution_decisions = existing_decisions + teds

    try:
        return agent.run(messages=messages, snapshot=snapshot.agent_snapshot if snapshot else None)
    except BreakpointException:
        return None


def run_pipeline_with_agent(
    pipeline: Pipeline,
    messages: list[ChatMessage],
    snapshot_file_path: Optional[str] = None,
    tool_execution_decisions: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    # Load the latest snapshot if a path is provided
    snapshot = None
    if snapshot_file_path:
        snapshot = get_latest_snapshot(snapshot_file_path=snapshot_file_path)

        # Add any new tool execution decisions to the snapshot
        if tool_execution_decisions:
            teds = [ToolExecutionDecision.from_dict(ted) for ted in tool_execution_decisions]
            existing_decisions = snapshot.agent_snapshot.tool_execution_decisions or []
            snapshot.agent_snapshot.tool_execution_decisions = existing_decisions + teds

    try:
        return pipeline.run({"agent": {"messages": messages}}, pipeline_snapshot=snapshot)
    except BreakpointException:
        return None


async def run_agent_async(
    agent: Agent,
    messages: list[ChatMessage],
    snapshot_file_path: Optional[str] = None,
    tool_execution_decisions: Optional[list[dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    # Load the latest snapshot if a path is provided
    snapshot = None
    if snapshot_file_path:
        snapshot = get_latest_snapshot(snapshot_file_path=snapshot_file_path)

        # Add any new tool execution decisions to the snapshot
        if tool_execution_decisions:
            teds = [ToolExecutionDecision.from_dict(ted) for ted in tool_execution_decisions]
            existing_decisions = snapshot.agent_snapshot.tool_execution_decisions or []
            snapshot.agent_snapshot.tool_execution_decisions = existing_decisions + teds

    try:
        return await agent.run_async(messages=messages, snapshot=snapshot.agent_snapshot if snapshot else None)
    except BreakpointException:
        return None


def addition_tool(a: int, b: int) -> int:
    return a + b


@pytest.fixture
def tools() -> list[Tool]:
    tool = create_tool_from_function(
        function=addition_tool, name="addition_tool", description="A tool that adds two integers together."
    )
    return [tool]


class TestAgent:
    def test_to_dict(self, tools, monkeypatch, mock_memory_client):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("MEM0_API_KEY", "test")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=tools, chat_message_store=InMemoryChatMessageStore(),
            memory_store=Mem0MemoryStore()
        )
        agent_dict = agent.to_dict()
        assert agent_dict == {
            "type": "haystack_experimental.components.agents.agent.Agent",
            "init_parameters": {
                "chat_generator": {
                    "type": "haystack.components.generators.chat.openai.OpenAIChatGenerator",
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "streaming_callback": None,
                        "api_base_url": None,
                        "organization": None,
                        "generation_kwargs": {},
                        "api_key": {"type": "env_var", "env_vars": ["OPENAI_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": None,
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                },
                "chat_message_store": {
                    "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore",
                    "init_parameters": {
                        "last_k": 10,
                        "skip_system_messages": True,
                    },
                },
                'memory_store': {
                    'type': 'haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore',
                    'init_parameters': {
                        'api_key':
                                {'type': 'env_var',
                                'env_vars':
                                    ['MEM0_API_KEY'],
                                    'strict': True}}},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "name": "addition_tool",
                            "description": "A tool that adds two integers together.",
                            "parameters": {
                                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                                "required": ["a", "b"],
                                "type": "object",
                            },
                            "function": "test.components.agents.test_agent.addition_tool",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    }
                ],
                "system_prompt": None,
                "exit_conditions": ["text"],
                "state_schema": {},
                "max_agent_steps": 100,
                "streaming_callback": None,
                "raise_on_tool_invocation_failure": False,
                "tool_invoker_kwargs": None,
                "confirmation_strategies": None,
            },
        }

    def test_from_dict(self, tools, monkeypatch, mock_memory_client):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        monkeypatch.setenv("MEM0_API_KEY", "test")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(), tools=tools, chat_message_store=InMemoryChatMessageStore(),
            memory_store=Mem0MemoryStore()
        )
        deserialized_agent = Agent.from_dict(agent.to_dict())
        assert deserialized_agent.to_dict() == agent.to_dict()
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        assert len(deserialized_agent.tools) == 1
        assert deserialized_agent.tools[0].name == "addition_tool"
        assert isinstance(deserialized_agent._tool_invoker, type(agent._tool_invoker))
        assert isinstance(deserialized_agent._chat_message_store, InMemoryChatMessageStore)
        assert isinstance(deserialized_agent._memory_store, Mem0MemoryStore)


class TestAgentConfirmationStrategy:
    def test_get_tool_calls_and_descriptions_from_snapshot_no_mutation_of_snapshot(
        self, tools, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED", "true")
        agent = Agent(
            chat_generator=MockChatGeneratorToolsResponse(),
            tools=tools,
            confirmation_strategies={
                "addition_tool": BreakpointConfirmationStrategy(snapshot_file_path=str(tmp_path)),
            },
        )
        agent.warm_up()

        # Run the agent to create a snapshot with a breakpoint
        try:
            agent.run([ChatMessage.from_user("What is 2+2?")])
        except BreakpointException:
            pass

        # Load the latest snapshot from disk
        loaded_snapshot = get_latest_snapshot(snapshot_file_path=str(tmp_path))

        original_snapshot = copy.deepcopy(loaded_snapshot)

        # Extract tool calls and descriptions
        _ = get_tool_calls_and_descriptions_from_snapshot(
            agent_snapshot=loaded_snapshot.agent_snapshot, breakpoint_tool_only=True
        )

        # Verify that the original snapshot has not been mutated
        assert loaded_snapshot == original_snapshot

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run_breakpoint_confirmation_strategy_modify(self, tools, tmp_path, monkeypatch):
        monkeypatch.setenv("HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED", "true")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            tools=tools,
            confirmation_strategies={
                "addition_tool": BreakpointConfirmationStrategy(snapshot_file_path=str(tmp_path)),
            },
        )
        agent.warm_up()

        # Step 1: Initial run
        result = run_agent(agent, [ChatMessage.from_user("What is 2+2?")])

        # Step 2: Loop to handle break point confirmation strategy until agent completes
        while result is None:
            # Load the latest snapshot from disk and prep data for front-end
            loaded_snapshot = get_latest_snapshot(snapshot_file_path=str(tmp_path))
            serialized_tool_calls, tool_descripts = get_tool_calls_and_descriptions_from_snapshot(
                agent_snapshot=loaded_snapshot.agent_snapshot, breakpoint_tool_only=True
            )

            # Simulate front-end interaction
            serialized_teds = frontend_simulate_tool_decision(
                serialized_tool_calls,
                tool_descripts,
                ConfirmationUIResult(action="modify", new_tool_params={"a": 2, "b": 3}),
            )

            # Re-run the agent with the new tool execution decisions
            result = run_agent(agent, [], str(tmp_path), serialized_teds)

        # Step 3: Final result
        last_message = result["last_message"]
        assert isinstance(last_message, ChatMessage)
        assert "5" in last_message.text

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run_in_pipeline_breakpoint_confirmation_strategy_modify(self, tools, tmp_path, monkeypatch):
        monkeypatch.setenv("HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED", "true")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            tools=tools,
            confirmation_strategies={
                "addition_tool": BreakpointConfirmationStrategy(snapshot_file_path=str(tmp_path)),
            },
        )

        pipeline = Pipeline()
        pipeline.add_component("agent", agent)

        # Step 1: Initial run
        result = run_pipeline_with_agent(pipeline, [ChatMessage.from_user("What is 2+2?")])

        # Step 2: Loop to handle break point confirmation strategy until pipeline with agent completes
        while result is None:
            # Load the latest snapshot from disk and prep data for front-end
            loaded_snapshot = get_latest_snapshot(snapshot_file_path=str(tmp_path))
            serialized_tool_calls, tool_descripts = get_tool_calls_and_descriptions_from_snapshot(
                agent_snapshot=loaded_snapshot.agent_snapshot, breakpoint_tool_only=True
            )

            # Simulate front-end interaction
            serialized_teds = frontend_simulate_tool_decision(
                serialized_tool_calls,
                tool_descripts,
                ConfirmationUIResult(action="modify", new_tool_params={"a": 2, "b": 3}),
            )

            # Re-run the agent with the new tool execution decisions
            result = run_pipeline_with_agent(pipeline, [], str(tmp_path), serialized_teds)

        # Step 3: Final result
        last_message = result["agent"]["last_message"]
        assert isinstance(last_message, ChatMessage)
        assert "5" in last_message.text

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_breakpoint_confirmation_strategy_modify(self, tools, tmp_path, monkeypatch):
        monkeypatch.setenv("HAYSTACK_PIPELINE_SNAPSHOT_SAVE_ENABLED", "true")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            tools=tools,
            confirmation_strategies={
                "addition_tool": BreakpointConfirmationStrategy(snapshot_file_path=str(tmp_path)),
            },
        )
        agent.warm_up()

        # Step 1: Initial run
        result = await run_agent_async(agent, [ChatMessage.from_user("What is 2+2?")])

        # Step 2: Loop to handle break point confirmation strategy until agent completes
        while result is None:
            # Load the latest snapshot from disk and prep data for front-end
            loaded_snapshot = get_latest_snapshot(snapshot_file_path=str(tmp_path))
            serialized_tool_calls, tool_descripts = get_tool_calls_and_descriptions_from_snapshot(
                agent_snapshot=loaded_snapshot.agent_snapshot, breakpoint_tool_only=True
            )

            # Simulate front-end interaction
            serialized_teds = frontend_simulate_tool_decision(
                serialized_tool_calls,
                tool_descripts,
                ConfirmationUIResult(action="modify", new_tool_params={"a": 2, "b": 3}),
            )

            # Re-run the agent with the new tool execution decisions
            result = await run_agent_async(agent, [], str(tmp_path), serialized_teds)

        # Step 3: Final result
        last_message = result["last_message"]
        assert isinstance(last_message, ChatMessage)
        assert "5" in last_message.text


class TestAgentWithChatMessageStore:
    def test_external_chat_message_store_with_agent(self, store):
        pipe = Pipeline()
        pipe.add_component(
            "prompt_builder",
            ChatPromptBuilder(template=[ChatMessage.from_user("{{ query }}")], required_variables=["query"]),
        )
        pipe.add_component("message_retriever", ChatMessageRetriever(store))
        pipe.add_component("agent", MockAgent(system_prompt="This is a system prompt."))
        pipe.add_component("message_writer", ChatMessageWriter(store))

        pipe.connect("prompt_builder.prompt", "message_retriever.current_messages")
        pipe.connect("message_retriever.messages", "agent.messages")
        pipe.connect("agent.messages", "message_writer.messages")

        chat_history_id = "user_123_session_456"
        result = pipe.run(
            data={
                "prompt_builder": {"query": "What is the capital of Germany?"},
                "message_retriever": {"chat_history_id": chat_history_id},
                "message_writer": {"chat_history_id": chat_history_id},
            },
            include_outputs_from={"agent"},
        )
        assert result["agent"]["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?"),
            ChatMessage.from_assistant("This is a mock response."),
        ]
        assert store.retrieve_messages(chat_history_id) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
        ]

        # Second run
        result = pipe.run(
            data={
                "prompt_builder": {"query": "What is the capital of Italy?"},
                "message_retriever": {"chat_history_id": chat_history_id},
                "message_writer": {"chat_history_id": chat_history_id},
            },
            include_outputs_from={"agent"},
        )
        assert result["agent"]["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Italy?"),
            ChatMessage.from_assistant("This is a mock response."),
        ]
        assert store.retrieve_messages(chat_history_id) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Italy?", meta={"chat_message_id": "2"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "3"}),
        ]

    def test_internal_chat_message_store_with_agent(self, store):
        agent = Agent(
            chat_generator=MockChatGenerator(), system_prompt="This is a system prompt.", chat_message_store=store
        )

        chat_history_id = "user_123_session_456"
        result = agent.run(
            messages=[ChatMessage.from_user("What is the capital of Germany?")],
            chat_message_store_kwargs={"chat_history_id": chat_history_id, "last_k": None},
        )
        assert result["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?"),
            ChatMessage.from_assistant("This is a mock response."),
        ]
        assert store.retrieve_messages(chat_history_id) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
        ]

        # Second run
        result = agent.run(
            messages=[ChatMessage.from_user("What is the capital of Italy?")],
            chat_message_store_kwargs={"chat_history_id": chat_history_id, "last_k": None},
        )
        assert result["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Italy?"),
            ChatMessage.from_assistant("This is a mock response."),
        ]
        assert store.retrieve_messages(chat_history_id) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Italy?", meta={"chat_message_id": "2"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "3"}),
        ]
