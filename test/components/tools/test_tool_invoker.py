# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.components.agents.human_in_the_loop import (
    ConfirmationStrategy,
    HumanInTheLoopStrategy,
    NeverAskPolicy,
    SimpleConsoleUI,
)
from haystack_experimental.components.tools import ToolInvoker


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
    return {"addition_tool": HumanInTheLoopStrategy(NeverAskPolicy(), SimpleConsoleUI())}


class TestToolInvoker:
    def test_to_dict(self, tools, confirmation_strategies):
        tool_invoker = ToolInvoker(tools=tools, confirmation_strategies=confirmation_strategies)
        tool_invoker_dict = tool_invoker.to_dict()
        assert tool_invoker_dict == {
            "type": "haystack_experimental.components.tools.tool_invoker.ToolInvoker",
            "init_parameters": {
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
                            "function": "test_tool_invoker.addition_tool",
                            "outputs_to_string": None,
                            "inputs_from_state": None,
                            "outputs_to_state": None,
                        },
                    }
                ],
                "raise_on_failure": True,
                "convert_result_to_json_string": False,
                "streaming_callback": None,
                "enable_streaming_callback_passthrough": False,
                "max_workers": 4,
                "confirmation_strategies": {
                    "addition_tool": {
                        "type": "haystack_experimental.components.agents.human_in_the_loop.strategies.HumanInTheLoopStrategy",
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

    def test_from_dict(self, tools, confirmation_strategies):
        tool_invoker = ToolInvoker(tools=tools, confirmation_strategies=confirmation_strategies)
        tool_invoker_dict = tool_invoker.to_dict()
        deserialized_tool_invoker = ToolInvoker.from_dict(tool_invoker_dict)
        assert isinstance(deserialized_tool_invoker, ToolInvoker)
        assert len(deserialized_tool_invoker.tools) == 1
        assert deserialized_tool_invoker.tools[0].name == "addition_tool"
        assert isinstance(deserialized_tool_invoker.confirmation_strategies["addition_tool"], HumanInTheLoopStrategy)
        assert isinstance(
            deserialized_tool_invoker.confirmation_strategies["addition_tool"].confirmation_policy, NeverAskPolicy
        )
        assert isinstance(
            deserialized_tool_invoker.confirmation_strategies["addition_tool"].confirmation_ui, SimpleConsoleUI
        )
