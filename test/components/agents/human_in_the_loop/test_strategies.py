# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack_experimental.components.agents.human_in_the_loop import (
    AskOncePolicy,
    BlockingConfirmationStrategy,
    BreakpointConfirmationStrategy,
    HITLBreakpointException,
    SimpleConsoleUI,
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
