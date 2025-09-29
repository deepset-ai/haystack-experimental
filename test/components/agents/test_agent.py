# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.components.agents.human_in_the_loop import (
    ConfirmationStrategy,
    HumanInTheLoopStrategy,
    NeverAskPolicy,
    SimpleConsoleUI,
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
def confirmation_strategies() -> dict[str, ConfirmationStrategy]:
    return {"addition_tool": HumanInTheLoopStrategy(NeverAskPolicy(), SimpleConsoleUI())}


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
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"], HumanInTheLoopStrategy)
        assert isinstance(
            deserialized_agent._confirmation_strategies["addition_tool"].confirmation_policy, NeverAskPolicy
        )
        assert isinstance(deserialized_agent._confirmation_strategies["addition_tool"].confirmation_ui, SimpleConsoleUI)
