# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import pytest
from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage, ToolCall, ChatRole, TextContent, ToolCallResult
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.components.agents.agent import _ExecutionContext
from haystack_experimental.components.agents.human_in_the_loop import (
    AlwaysAskPolicy,
    AskOncePolicy,
    BlockingConfirmationStrategy,
    BreakpointConfirmationStrategy,
    HITLBreakpointException,
    SimpleConsoleUI,
    ToolExecutionDecision,
    NeverAskPolicy,
    ConfirmationUIResult,
)
from haystack_experimental.components.agents.human_in_the_loop.strategies import (
    _apply_tool_execution_decisions,
    _run_confirmation_strategies,
    _update_chat_history,
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

    def test_run_confirm(self, monkeypatch):
        strategy = BlockingConfirmationStrategy(AlwaysAskPolicy(), SimpleConsoleUI())

        # Mock the UI to always confirm
        def mock_get_user_confirmation(tool_name, tool_description, tool_params):
            return ConfirmationUIResult(action="confirm")

        monkeypatch.setattr(strategy.confirmation_ui, "get_user_confirmation", mock_get_user_confirmation)

        decision = strategy.run(tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"})
        assert decision.tool_name == "test_tool"
        assert decision.execute is True
        assert decision.final_tool_params == {"param1": "value1"}

    def test_run_modify(self, monkeypatch):
        strategy = BlockingConfirmationStrategy(AlwaysAskPolicy(), SimpleConsoleUI())

        # Mock the UI to always modify
        def mock_get_user_confirmation(tool_name, tool_description, tool_params):
            return ConfirmationUIResult(action="modify", new_tool_params={"param1": "new_value"})

        monkeypatch.setattr(strategy.confirmation_ui, "get_user_confirmation", mock_get_user_confirmation)

        decision = strategy.run(tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"})
        assert decision.tool_name == "test_tool"
        assert decision.execute is True
        assert decision.final_tool_params == {"param1": "new_value"}
        assert decision.feedback == (
            "The parameters for tool 'test_tool' were updated by the user to:\n{'param1': 'new_value'}"
        )

    def test_run_reject(self, monkeypatch):
        strategy = BlockingConfirmationStrategy(AlwaysAskPolicy(), SimpleConsoleUI())

        # Mock the UI to always reject
        def mock_get_user_confirmation(tool_name, tool_description, tool_params):
            return ConfirmationUIResult(action="reject", feedback="Not needed")

        monkeypatch.setattr(strategy.confirmation_ui, "get_user_confirmation", mock_get_user_confirmation)

        decision = strategy.run(tool_name="test_tool", tool_description="A test tool", tool_params={"param1": "value1"})
        assert decision.tool_name == "test_tool"
        assert decision.execute is False
        assert decision.final_tool_params is None
        assert decision.feedback == "Tool execution for 'test_tool' was rejected by the user. With feedback: Not needed"


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
                ToolExecutionDecision(
                    tool_name=tools[0].name,
                    execute=False,
                    tool_call_id="1",
                    feedback=(
                        "The tool execution for 'addition_tool' was rejected by the user. With feedback: Not needed"
                    ),
                )
            ],
        )
        assert rejection_messages == [
            assistant_message,
            ChatMessage.from_tool(
                tool_result=(
                    "The tool execution for 'addition_tool' was rejected by the user. With feedback: Not needed"
                ),
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
                    tool_name=tools[0].name,
                    execute=True,
                    tool_call_id="1",
                    final_tool_params={"a": 5, "b": 6},
                    feedback="The parameters for tool 'addition_tool' were updated by the user to:\n{'a': 5, 'b': 6}",
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


class TestUpdateChatHistory:
    @pytest.fixture
    def chat_history_one_tool_call(self):
        return [
            ChatMessage.from_user("Hello"),
            ChatMessage.from_assistant(tool_calls=[ToolCall("tool1", {"a": 1, "b": 2}, id="1")]),
        ]

    def test_update_chat_history_rejection(self, chat_history_one_tool_call):
        """Test that the new history includes a tool call result message after a rejection."""
        rejection_messages = [
            ChatMessage.from_assistant(tool_calls=[chat_history_one_tool_call[1].tool_call]),
            ChatMessage.from_tool(
                tool_result="The tool execution for 'tool1' was rejected by the user. With feedback: Not needed",
                origin=chat_history_one_tool_call[1].tool_call,
                error=True,
            ),
        ]
        updated_messages = _update_chat_history(
            chat_history_one_tool_call, rejection_messages=rejection_messages, tool_call_and_explanation_messages=[]
        )
        assert updated_messages == [chat_history_one_tool_call[0], *rejection_messages]

    def test_update_chat_history_confirm(self, chat_history_one_tool_call):
        """No changes should be made if the tool call was confirmed."""
        tool_call_messages = [ChatMessage.from_assistant(tool_calls=[chat_history_one_tool_call[1].tool_call])]
        updated_messages = _update_chat_history(
            chat_history_one_tool_call, rejection_messages=[], tool_call_and_explanation_messages=tool_call_messages
        )
        assert updated_messages == chat_history_one_tool_call

    def test_update_chat_history_modify(self, chat_history_one_tool_call):
        """Test that the new history includes a user message and updated tool call after a modification."""
        tool_call_messages = [
            ChatMessage.from_user(
                "The parameters for tool 'tool1' were updated by the user to:\n{'param': 'new_value'}"
            ),
            ChatMessage.from_assistant(tool_calls=[ToolCall("tool1", {"param": "new_value"}, id="1")]),
        ]
        updated_messages = _update_chat_history(
            chat_history_one_tool_call, rejection_messages=[], tool_call_and_explanation_messages=tool_call_messages
        )
        assert updated_messages == [chat_history_one_tool_call[0], *tool_call_messages]

    def test_update_chat_history_modify_two_tool_calls(self):
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall("tool1", {"a": 1, "b": 2}, id="1"), ToolCall("tool2", {"a": 3, "b": 4}, id="2")]
        )
        chat_history = [ChatMessage.from_user("What is 1 + 2? and 3 + 4?"), tool_call_message]
        rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[tool_call_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name="tool1",
                    execute=True,
                    tool_call_id="1",
                    final_tool_params={"a": 5, "b": 6},
                    feedback="The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}",
                ),
                ToolExecutionDecision(
                    tool_name="tool2",
                    execute=True,
                    tool_call_id="2",
                    final_tool_params={"a": 7, "b": 8},
                    feedback="The parameters for tool 'tool2' were updated by the user to:\n{'a': 7, 'b': 8}'",
                ),
            ],
        )
        updated_messages = _update_chat_history(
            chat_history,
            rejection_messages=rejection_messages,
            tool_call_and_explanation_messages=modified_tool_call_messages,
        )
        assert updated_messages == [
            chat_history[0],
            ChatMessage.from_user("The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}"),
            ChatMessage.from_user("The parameters for tool 'tool2' were updated by the user to:\n{'a': 7, 'b': 8}'"),
            ChatMessage.from_assistant(
                tool_calls=[
                    ToolCall(tool_name="tool1", arguments={"a": 5, "b": 6}, id="1", extra=None),
                    ToolCall(tool_name="tool2", arguments={"a": 7, "b": 8}, id="2", extra=None),
                ],
            ),
        ]

    def test_update_chat_history_two_tool_calls_modify_and_reject(self):
        tool_call_message = ChatMessage.from_assistant(
            tool_calls=[ToolCall("tool1", {"a": 1, "b": 2}, id="1"), ToolCall("tool2", {"a": 3, "b": 4}, id="2")]
        )
        chat_history = [ChatMessage.from_user("What is 1 + 2? and 3 + 4?"), tool_call_message]
        rejection_messages, modified_tool_call_messages = _apply_tool_execution_decisions(
            tool_call_messages=[tool_call_message],
            tool_execution_decisions=[
                ToolExecutionDecision(
                    tool_name="tool1",
                    execute=True,
                    tool_call_id="1",
                    final_tool_params={"a": 5, "b": 6},
                    feedback="The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}",
                ),
                ToolExecutionDecision(
                    tool_name="tool2",
                    execute=False,
                    tool_call_id="2",
                    feedback="The tool execution for 'tool2' was rejected by the user. With feedback: Not needed",
                ),
            ],
        )
        updated_messages = _update_chat_history(
            chat_history,
            rejection_messages=rejection_messages,
            tool_call_and_explanation_messages=modified_tool_call_messages,
        )
        assert updated_messages == [
            chat_history[0],
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="tool2", arguments={"a": 3, "b": 4}, id="2", extra=None)],
            ),
            ChatMessage.from_tool(
                tool_result="The tool execution for 'tool2' was rejected by the user. With feedback: Not needed",
                origin=ToolCall(tool_name="tool2", arguments={"a": 3, "b": 4}, id="2", extra=None),
                error=True,
            ),
            ChatMessage.from_user("The parameters for tool 'tool1' were updated by the user to:\n{'a': 5, 'b': 6}"),
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(tool_name="tool1", arguments={"a": 5, "b": 6}, id="1", extra=None)],
            ),
        ]
