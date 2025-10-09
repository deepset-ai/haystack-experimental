# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, Optional

import pytest
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import PipelineSnapshot
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.components.agents.human_in_the_loop import (
    AlwaysAskPolicy,
    BlockingConfirmationStrategy,
    BreakpointConfirmationStrategy,
    ConfirmationStrategy,
    ConfirmationUI,
    ConfirmationUIResult,
    NeverAskPolicy,
    SimpleConsoleUI,
    ToolExecutionDecision,
)
from haystack_experimental.components.agents.human_in_the_loop.breakpoint import (
    get_tool_calls_and_descriptions_from_snapshot,
)


class TestUserInterface(ConfirmationUI):
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
        confirmation_ui=TestUserInterface(ui_result=confirmation_ui_result),
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


def addition_tool(a: int, b: int) -> int:
    return a + b


@pytest.fixture
def tools() -> list[Tool]:
    tool = create_tool_from_function(
        function=addition_tool, name="addition_tool", description="A tool that adds two integers together."
    )
    return [tool]


@pytest.fixture
def confirmation_strategies() -> dict[str, ConfirmationStrategy]:
    return {"addition_tool": BlockingConfirmationStrategy(NeverAskPolicy(), SimpleConsoleUI())}


class TestAgent:
    def test_to_dict(self, tools, confirmation_strategies, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(), tools=tools, confirmation_strategies=confirmation_strategies
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
                "confirmation_strategies": {
                    "addition_tool": {
                        "type": "haystack_experimental.components.agents.human_in_the_loop.strategies.BlockingConfirmationStrategy",
                        "init_parameters": {
                            "confirmation_policy": {
                                "type": "haystack_experimental.components.agents.human_in_the_loop.policies.NeverAskPolicy",
                                "init_parameters": {},
                            },
                            "confirmation_ui": {
                                "type": "haystack_experimental.components.agents.human_in_the_loop.user_interfaces.SimpleConsoleUI",
                                "init_parameters": {},
                            },
                        },
                    }
                },
            },
        }

    def test_from_dict(self, tools, confirmation_strategies, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        agent = Agent(
            chat_generator=OpenAIChatGenerator(), tools=tools, confirmation_strategies=confirmation_strategies
        )
        deserialized_agent = Agent.from_dict(agent.to_dict())
        assert deserialized_agent.to_dict() == agent.to_dict()
        assert isinstance(deserialized_agent.chat_generator, OpenAIChatGenerator)
        assert len(deserialized_agent.tools) == 1
        assert deserialized_agent.tools[0].name == "addition_tool"
        assert isinstance(deserialized_agent._tool_invoker, type(agent._tool_invoker))
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"], BlockingConfirmationStrategy)
        assert isinstance(
            deserialized_agent._confirmation_strategies["addition_tool"].confirmation_policy, NeverAskPolicy
        )
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"].confirmation_ui, SimpleConsoleUI)


class TestAgentConfirmationStrategy:
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run_blocking_confirmation_strategy_modify(self, tools):
        agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
            tools=tools,
            confirmation_strategies={
                "addition_tool": BlockingConfirmationStrategy(
                    AlwaysAskPolicy(),
                    TestUserInterface(ConfirmationUIResult(action="modify", new_tool_params={"a": 2, "b": 3})),
                )
            },
        )
        agent.warm_up()

        result = agent.run([ChatMessage.from_user("What is 2+2?")])

        assert isinstance(result["last_message"], ChatMessage)
        assert "5" in result["last_message"].text

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_run_breakpoint_confirmation_strategy_modify(self, tools, tmp_path):
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
