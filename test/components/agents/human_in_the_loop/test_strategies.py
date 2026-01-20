# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import pytest
from haystack.human_in_the_loop.strategies import (
    _run_confirmation_strategies,
    _run_confirmation_strategies_async,
)
from haystack.components.agents.agent import _ExecutionContext
from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.components.agents.human_in_the_loop import (
    BreakpointConfirmationStrategy,
    HITLBreakpointException,
)


def addition_tool(a: int, b: int) -> int:
    return a + b


@pytest.fixture
def tools() -> list[Tool]:
    tool = create_tool_from_function(
        function=addition_tool, name="addition_tool", description="A tool that adds two integers together."
    )
    return [tool]


@pytest.fixture
def execution_context(tools: list[Tool]) -> _ExecutionContext:
    return _ExecutionContext(
        state=State(schema={"messages": {"type": list[ChatMessage]}}),
        component_visits={"chat_generator": 0, "tool_invoker": 0},
        chat_generator_inputs={},
        tool_invoker_inputs={"tools": tools},
        counter=0,
        skip_chat_generator=False,
        tool_execution_decisions=None,
    )


class TestBreakpointConfirmationStrategy:
    def test_initialization(self):
        strategy = BreakpointConfirmationStrategy(snapshot_file_path="test")
        assert strategy.snapshot_file_path == "test"

    def test_to_dict(self):
        strategy = BreakpointConfirmationStrategy(snapshot_file_path="test")
        strategy_dict = strategy.to_dict()
        assert strategy_dict == {
            "type": "haystack_experimental.components.agents.human_in_the_loop.strategies.BreakpointConfirmationStrategy",
            "init_parameters": {"snapshot_file_path": "test"},
        }

    def test_from_dict(self):
        strategy_dict = {
            "type": "haystack_experimental.components.agents.human_in_the_loop.strategies.BreakpointConfirmationStrategy",
            "init_parameters": {"snapshot_file_path": "test"},
        }
        strategy = BreakpointConfirmationStrategy.from_dict(strategy_dict)
        assert isinstance(strategy, BreakpointConfirmationStrategy)
        assert strategy.snapshot_file_path == "test"

    def test_run(self):
        strategy = BreakpointConfirmationStrategy(snapshot_file_path="test")
        with pytest.raises(HITLBreakpointException):
            strategy.run(tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"})

    def test_run_confirmation_strategies_hitl_breakpoint(self, tmp_path, tools, execution_context):
        with pytest.raises(HITLBreakpointException):
            _run_confirmation_strategies(
                confirmation_strategies={tools[0].name: BreakpointConfirmationStrategy(str(tmp_path))},
                messages_with_tool_calls=[
                    ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"})]),
                ],
                execution_context=execution_context,
            )


class TestAsyncConfirmationStrategies:
    @pytest.mark.asyncio
    async def test_breakpoint_strategy_run_async(self):
        strategy = BreakpointConfirmationStrategy(snapshot_file_path="test_path")

        with pytest.raises(HITLBreakpointException) as exc_info:
            await strategy.run_async(
                tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"}
            )

        assert exc_info.value.tool_name == "test_tool"
        assert exc_info.value.snapshot_file_path == "test_path"

    @pytest.mark.asyncio
    async def test_run_confirmation_strategies_async_hitl_breakpoint(self, tmp_path, tools, execution_context):
        with pytest.raises(HITLBreakpointException):
            await _run_confirmation_strategies_async(
                confirmation_strategies={tools[0].name: BreakpointConfirmationStrategy(str(tmp_path))},
                messages_with_tool_calls=[
                    ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"a": 1, "b": 2})]),
                ],
                execution_context=execution_context,
            )
