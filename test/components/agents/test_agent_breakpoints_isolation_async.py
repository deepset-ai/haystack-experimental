# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool

from haystack_experimental.components.agents import Agent
from haystack_experimental.core.errors import AgentBreakpointException
from haystack_experimental.core.pipeline.breakpoint import load_state

from test.components.agents.test_agent import MockChatGeneratorWithRunAsync, weather_function


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
    generator = MockChatGeneratorWithRunAsync()
    mock_run_async = AsyncMock()
    mock_run_async.return_value = {
        "replies": [
            ChatMessage.from_assistant("I'll help you check the weather.", tool_calls=[{
                "tool_name": "weather_tool",
                "tool_args": {"location": "Berlin"}
            }])
        ]
    }
    async def mock_run_async_with_tools(messages, tools=None, **kwargs):
        return mock_run_async.return_value
    generator.run_async = mock_run_async_with_tools
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


@pytest.fixture
def mock_agent_with_tool_calls(monkeypatch, weather_tool):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    generator = MockChatGeneratorWithRunAsync()
    mock_messages = [
        ChatMessage.from_assistant("First response"),
        ChatMessage.from_assistant(
            tool_calls=[ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"})]
        ),
    ]
    agent = Agent(chat_generator=generator, tools=[weather_tool], max_agent_steps=1)
    agent.warm_up()
    agent.chat_generator.run_async = AsyncMock(return_value={"replies": mock_messages})
    return agent


@pytest.mark.asyncio
async def test_run_async_with_chat_generator_breakpoint(agent, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {("chat_generator", 0, None)}
    with pytest.raises(AgentBreakpointException) as exc_info:
        await agent.run_async(messages=messages, agent_breakpoints=breakpoints, debug_path=debug_path)
    assert exc_info.value.component == "chat_generator"
    assert "messages" in exc_info.value.state


@pytest.mark.asyncio
async def test_run_async_with_tool_invoker_breakpoint(mock_agent_with_tool_calls, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {("tool_invoker", 0, "weather_tool")}
    with pytest.raises(AgentBreakpointException) as exc_info:
        await mock_agent_with_tool_calls.run_async(messages=messages, agent_breakpoints=breakpoints, debug_path=debug_path)

    assert exc_info.value.component == "tool_invoker"
    assert "messages" in exc_info.value.state


@pytest.mark.asyncio
async def test_resume_from_chat_generator_async(agent, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {("chat_generator", 0, None)}
    
    try:
        await agent.run_async(messages=messages, agent_breakpoints=breakpoints, debug_path=debug_path)
    except AgentBreakpointException:
        pass

    state_files = list(Path(debug_path).glob("chat_generator_*.json"))
    assert len(state_files) > 0
    latest_state_file = str(max(state_files, key=os.path.getctime))

    resume_state = load_state(latest_state_file)
    result = await agent.run_async(
        messages=[ChatMessage.from_user("Continue from where we left off.")],
        resume_state=resume_state
    )

    assert "messages" in result
    assert "last_message" in result
    assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_resume_from_tool_invoker_async(mock_agent_with_tool_calls, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {("tool_invoker", 0, "weather_tool")}
    
    try:
        await mock_agent_with_tool_calls.run_async(messages=messages, agent_breakpoints=breakpoints, debug_path=debug_path)
    except AgentBreakpointException:
        pass

    state_files = list(Path(debug_path).glob("tool_invoker_*.json"))
    assert len(state_files) > 0
    latest_state_file = str(max(state_files, key=os.path.getctime))

    resume_state = load_state(latest_state_file)

    result = await mock_agent_with_tool_calls.run_async(
        messages=[ChatMessage.from_user("Continue from where we left off.")],
        resume_state=resume_state
    )

    assert "messages" in result
    assert "last_message" in result
    assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_invalid_combination_breakpoint_and_resume_state_async(mock_agent_with_tool_calls, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {("tool_invoker", 0, "weather_tool")}
    with pytest.raises(ValueError, match="agent_breakpoint and resume_state cannot be provided at the same time"):
        await mock_agent_with_tool_calls.run_async(
            messages=messages, 
            agent_breakpoints=breakpoints, 
            debug_path=debug_path, 
            resume_state={"some": "state"}
        )


@pytest.mark.asyncio
async def test_breakpoint_with_invalid_component_async(mock_agent_with_tool_calls, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {("invalid_breakpoint", 0, "weather_tool")}
    with pytest.raises(ValueError, match="Breakpoint 'invalid_breakpoint' is not a valid component."):
        await mock_agent_with_tool_calls.run_async(messages=messages, agent_breakpoints=breakpoints, debug_path=debug_path)


@pytest.mark.asyncio
async def test_breakpoint_with_invalid_tool_name_async(mock_agent_with_tool_calls, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {("tool_invoker", 0, "invalid_tool")}
    with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
        await mock_agent_with_tool_calls.run_async(messages=messages, agent_breakpoints=breakpoints, debug_path=debug_path)


@pytest.mark.asyncio
async def test_run_async_with_multiple_breakpoints(agent, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    breakpoints = {
        ("chat_generator", 0, None),
        ("tool_invoker", 0, "weather_tool")
    }
    with pytest.raises(AgentBreakpointException) as exc_info:
        await agent.run_async(messages=messages, agent_breakpoints=breakpoints, debug_path=debug_path)

    # break is at the first breakpoint (chat_generator)
    assert exc_info.value.component == "chat_generator"
