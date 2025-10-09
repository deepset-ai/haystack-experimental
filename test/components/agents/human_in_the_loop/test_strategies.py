# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import pytest
from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.components.agents.agent import _ExecutionContext
from haystack_experimental.components.agents.human_in_the_loop import (
    AskOncePolicy,
    BlockingConfirmationStrategy,
    BreakpointConfirmationStrategy,
    HITLBreakpointException,
    SimpleConsoleUI,
    ToolExecutionDecision,
    NeverAskPolicy,
)
from haystack_experimental.components.agents.human_in_the_loop.strategies import (
    _apply_tool_execution_decisions,
    _run_confirmation_strategies,
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
def execution_context(tools) -> _ExecutionContext:
    return _ExecutionContext(
        state=State(schema={"messages": {"type": list[ChatMessage]}}),
        component_visits={"chat_generator": 0, "tool_invoker": 0},
        chat_generator_inputs={},
        tool_invoker_inputs={"tools": tools},
        counter=0,
        skip_chat_generator=False,
        tool_execution_decisions=None,
    )


class TestBlockingConfirmationStrategy:
    def test_initialization(self):
        strategy = BlockingConfirmationStrategy(confirmation_policy=AskOncePolicy(), confirmation_ui=SimpleConsoleUI())
        assert isinstance(strategy.confirmation_policy, AskOncePolicy)
        assert isinstance(strategy.confirmation_ui, SimpleConsoleUI)

    def test_to_dict(self):
        strategy = BlockingConfirmationStrategy(confirmation_policy=AskOncePolicy(), confirmation_ui=SimpleConsoleUI())
        strategy_dict = strategy.to_dict()
        assert strategy_dict == {
            "type": "haystack_experimental.components.agents.human_in_the_loop.strategies.BlockingConfirmationStrategy",
            "init_parameters": {
                "confirmation_policy": {
                    "type": "haystack_experimental.components.agents.human_in_the_loop.policies.AskOncePolicy",
                    "init_parameters": {},
                },
                "confirmation_ui": {
                    "type": "haystack_experimental.components.agents.human_in_the_loop.user_interfaces.SimpleConsoleUI",
                    "init_parameters": {},
                },
            },
        }

    def test_from_dict(self):
        strategy_dict = {
            "type": "haystack_experimental.components.agents.human_in_the_loop.strategies.HumanInTheLoopStrategy",
            "init_parameters": {
                "confirmation_policy": {
                    "type": "haystack_experimental.components.agents.human_in_the_loop.policies.AskOncePolicy",
                    "init_parameters": {},
                },
                "confirmation_ui": {
                    "type": "haystack_experimental.components.agents.human_in_the_loop.user_interfaces.SimpleConsoleUI",
                    "init_parameters": {},
                },
            },
        }
        strategy = BlockingConfirmationStrategy.from_dict(strategy_dict)
        assert isinstance(strategy, BlockingConfirmationStrategy)
        assert isinstance(strategy.confirmation_policy, AskOncePolicy)
        assert isinstance(strategy.confirmation_ui, SimpleConsoleUI)


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


class TestRunConfirmationStrategies:
    def test_run_confirmation_strategies_hitl_breakpoint(self, tmp_path, tools, execution_context):
        with pytest.raises(HITLBreakpointException):
            _run_confirmation_strategies(
                confirmation_strategies={tools[0].name: BreakpointConfirmationStrategy(str(tmp_path))},
                messages_with_tool_calls=[
                    ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"})]),
                ],
                execution_context=execution_context,
            )

    def test_run_confirmation_strategies_no_strategy(self, tools, execution_context):
        teds = _run_confirmation_strategies(
            confirmation_strategies={},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"})]),
            ],
            execution_context=execution_context,
        )
        assert teds == [
            ToolExecutionDecision(tool_name=tools[0].name, execute=True, final_tool_params={"param1": "value1"})
        ]

    def test_run_confirmation_strategies_with_strategy(self, tools, execution_context):
        teds = _run_confirmation_strategies(
            confirmation_strategies={tools[0].name: BlockingConfirmationStrategy(NeverAskPolicy(), SimpleConsoleUI())},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"})]),
            ],
            execution_context=execution_context,
        )
        assert teds == [
            ToolExecutionDecision(tool_name=tools[0].name, execute=True, final_tool_params={"param1": "value1"})
        ]

    def test_run_confirmation_strategies_with_existing_teds(self, tools, execution_context):
        exe_context_with_teds = replace(
            execution_context,
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name, execute=True, tool_call_id="123", final_tool_params={"param1": "new_value"}
                )
            ],
        )
        teds = _run_confirmation_strategies(
            confirmation_strategies={tools[0].name: BlockingConfirmationStrategy(NeverAskPolicy(), SimpleConsoleUI())},
            messages_with_tool_calls=[
                ChatMessage.from_assistant(tool_calls=[ToolCall(tools[0].name, {"param1": "value1"}, id="123")]),
            ],
            execution_context=exe_context_with_teds,
        )
        assert teds == [
            ToolExecutionDecision(
                tool_name=tools[0].name, execute=True, tool_call_id="123", final_tool_params={"param1": "new_value"}
            )
        ]


class TestApplyToolExecutionDecisions:
    @pytest.fixture
    def assistant_message(self, tools):
        tool_call = ToolCall(tool_name=tools[0].name, arguments={"a": 1, "b": 2}, id="1")
        return ChatMessage.from_assistant(tool_calls=[tool_call])

    def test_apply_tool_execution_decisions_reject(self, tools, assistant_message):
        rejection_messages, new_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[assistant_message],
            tool_execution_decisions=[
                ToolExecutionDecision(tool_name=tools[0].name, execute=False, tool_call_id="1", feedback="Not needed")
            ],
        )
        assert rejection_messages == [
            assistant_message,
            ChatMessage.from_tool(
                tool_result="Tool execution for 'addition_tool' was rejected by the user. Feedback: Not needed",
                origin=assistant_message.tool_call,
                error=True,
            ),
        ]
        assert new_tool_call_messages == []

    def test_apply_tool_execution_decisions_confirm(self, tools, assistant_message):
        rejection_messages, new_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[assistant_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name, execute=True, tool_call_id="1", final_tool_params={"a": 1, "b": 2}
                )
            ],
        )
        assert rejection_messages == []
        assert new_tool_call_messages == [assistant_message]

    def test_apply_tool_execution_decisions_modify(self, tools, assistant_message):
        rejection_messages, new_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[assistant_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name=tools[0].name, execute=True, tool_call_id="1", final_tool_params={"a": 5, "b": 6}
                )
            ],
        )
        assert rejection_messages == []
        assert new_tool_call_messages == [
            ChatMessage.from_user(
                "The parameters for tool 'addition_tool' were updated by the user to:\n{'a': 5, 'b': 6}"
            ),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name=tools[0].name, arguments={"a": 5, "b": 6}, id="1")]
            ),
        ]
