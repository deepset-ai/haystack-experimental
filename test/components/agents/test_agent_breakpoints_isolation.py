# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from haystack.dataclasses import ChatMessage
from haystack.tools import Tool

from haystack_experimental.components.agents import Agent
from haystack_experimental.core.errors import AgentBreakpointException
from haystack_experimental.core.pipeline.breakpoint import load_state

from test.components.agents.test_agent import (
    MockChatGeneratorWithoutRunAsync,
    weather_function,
)


@pytest.fixture
def weather_tool():
    return Tool(
        name="weather_tool",
        description="Provides weather information for a given location.",
        parameters={"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        function=weather_function,
    )


@pytest.fixture
def mock_chat_generator():
    generator = MockChatGeneratorWithoutRunAsync()
    mock_run = MagicMock()
    mock_run.return_value = {
        "replies": [
            ChatMessage.from_assistant("I'll help you check the weather.", tool_calls=[{
                "tool_name": "weather_tool",
                "tool_args": {"location": "Berlin"}
            }])
        ]
    }
    def mock_run_with_tools(messages, tools=None, **kwargs):
        return mock_run.return_value
    generator.run = mock_run_with_tools
    return generator


@pytest.fixture
def agent(mock_chat_generator, weather_tool):
    return Agent(
        chat_generator=mock_chat_generator,
        tools=[weather_tool],
        system_prompt="You are a helpful assistant that can use tools to help users.",
    )


@pytest.fixture
def debug_path(tmp_path):
    return str(tmp_path / "debug_states")


def test_run_with_chat_generator_breakpoint(agent, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoint = ("chat_generator", 0, None)
    with pytest.raises(AgentBreakpointException) as exc_info:
        agent.run(messages=messages, agent_breakpoint=breakpoint, debug_path=debug_path)
    assert exc_info.value.component == "chat_generator"
    assert "messages" in exc_info.value.state


def test_run_with_tool_invoker_breakpoint(agent, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoint = ("tool_invoker", 0, "weather_tool")
    with pytest.raises(AgentBreakpointException) as exc_info:
        agent.run(messages=messages, agent_breakpoint=breakpoint, debug_path=debug_path)
    assert exc_info.value.component == "tool_invoker"
    assert "messages" in exc_info.value.state


def test_resume_from_saved_state(agent, debug_path):
    """Test resuming the agent from a saved state."""
    # First run to create a state file
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoint = ("chat_generator", 0, None)
    
    try:
        agent.run(messages=messages, agent_breakpoint=breakpoint, debug_path=debug_path)
    except AgentBreakpointException:
        pass
    
    # Find the most recent state file
    state_files = list(Path(debug_path).glob("chat_generator_*.json"))
    assert len(state_files) > 0
    latest_state_file = str(max(state_files, key=os.path.getctime))
    
    # Load the state and resume
    resume_state = load_state(latest_state_file)
    result = agent.run(
        messages=[ChatMessage.from_user("Continue from where we left off.")],
        resume_state=resume_state
    )
    
    assert "messages" in result
    assert "last_message" in result
    assert len(result["messages"]) > 0


def test_invalid_breakpoint_combination(agent):
    """Test that providing both breakpoint and resume_state raises an error."""
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoint = ("chat_generator", 0, None)
    resume_state = {"some": "state"}
    
    with pytest.raises(ValueError, match="agent_breakpoint and resume_state cannot be provided at the same time"):
        agent.run(
            messages=messages,
            agent_breakpoint=breakpoint,
            resume_state=resume_state
        )


def test_breakpoint_with_invalid_component(agent, debug_path):
    """Test that providing an invalid component name in breakpoint raises an error."""
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoint = ("invalid_component", 0, None)
    
    with pytest.raises(ValueError):
        agent.run(messages=messages, agent_breakpoint=breakpoint, debug_path=debug_path)


def test_breakpoint_with_invalid_tool_name(agent, debug_path):
    """Test that providing an invalid tool name in breakpoint raises an error."""
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoint = ("tool_invoker", 0, "invalid_tool")
    
    with pytest.raises(ValueError):
        agent.run(messages=messages, agent_breakpoint=breakpoint, debug_path=debug_path)
