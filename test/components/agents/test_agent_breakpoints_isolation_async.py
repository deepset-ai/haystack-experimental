# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from pathlib import Path

from haystack.dataclasses import ChatMessage

from haystack_experimental.core.errors import AgentBreakpointException
from haystack_experimental.core.pipeline.breakpoint import load_state
from haystack_experimental.dataclasses.breakpoints import AgentBreakpoint, Breakpoint

from test.components.agents.test_agent_breakpoints_utils import (
    create_chat_generator_breakpoint,
    create_tool_breakpoint,
    create_agent_breakpoint,
    weather_tool,
    debug_path,
    agent_async,
    mock_agent_with_tool_calls_async,
)


@pytest.mark.asyncio
async def test_run_async_with_chat_generator_breakpoint(agent_async, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    chat_generator_bp = create_chat_generator_breakpoint(visit_count=0)
    agent_breakpoint = create_agent_breakpoint(chat_generator_breakpoints={chat_generator_bp})
    with pytest.raises(AgentBreakpointException) as exc_info:
        await agent_async.run_async(messages=messages, agent_breakpoints=agent_breakpoint, debug_path=debug_path)
    assert exc_info.value.component == "chat_generator"
    assert "messages" in exc_info.value.state


@pytest.mark.asyncio
async def test_run_async_with_tool_invoker_breakpoint(mock_agent_with_tool_calls_async, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = create_tool_breakpoint(tool_name="weather_tool", visit_count=0)
    agent_breakpoint = create_agent_breakpoint(tool_breakpoints={tool_bp})
    with pytest.raises(AgentBreakpointException) as exc_info:
        await mock_agent_with_tool_calls_async.run_async(messages=messages, agent_breakpoints=agent_breakpoint, debug_path=debug_path)

    assert exc_info.value.component == "tool_invoker"
    assert "messages" in exc_info.value.state


@pytest.mark.asyncio
async def test_resume_from_chat_generator_async(agent_async, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    chat_generator_bp = create_chat_generator_breakpoint(visit_count=0)
    agent_breakpoint = create_agent_breakpoint(chat_generator_breakpoints={chat_generator_bp})
    
    try:
        await agent_async.run_async(messages=messages, agent_breakpoints=agent_breakpoint, debug_path=debug_path)
    except AgentBreakpointException:
        pass

    state_files = list(Path(debug_path).glob("chat_generator_*.json"))
    assert len(state_files) > 0
    latest_state_file = str(max(state_files, key=os.path.getctime))

    resume_state = load_state(latest_state_file)
    result = await agent_async.run_async(
        messages=[ChatMessage.from_user("Continue from where we left off.")],
        resume_state=resume_state
    )

    assert "messages" in result
    assert "last_message" in result
    assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_resume_from_tool_invoker_async(mock_agent_with_tool_calls_async, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = create_tool_breakpoint(tool_name="weather_tool", visit_count=0)
    agent_breakpoint = create_agent_breakpoint(tool_breakpoints={tool_bp})
    
    try:
        await mock_agent_with_tool_calls_async.run_async(messages=messages, agent_breakpoints=agent_breakpoint, debug_path=debug_path)
    except AgentBreakpointException:
        pass

    state_files = list(Path(debug_path).glob("tool_invoker_*.json"))
    assert len(state_files) > 0
    latest_state_file = str(max(state_files, key=os.path.getctime))

    resume_state = load_state(latest_state_file)

    result = await mock_agent_with_tool_calls_async.run_async(
        messages=[ChatMessage.from_user("Continue from where we left off.")],
        resume_state=resume_state
    )

    assert "messages" in result
    assert "last_message" in result
    assert len(result["messages"]) > 0


@pytest.mark.asyncio
async def test_invalid_combination_breakpoint_and_resume_state_async(mock_agent_with_tool_calls_async, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = create_tool_breakpoint(tool_name="weather_tool", visit_count=0)
    agent_breakpoint = create_agent_breakpoint(tool_breakpoints={tool_bp})
    with pytest.raises(ValueError, match="agent_breakpoint and resume_state cannot be provided at the same time"):
        await mock_agent_with_tool_calls_async.run_async(
            messages=messages, 
            agent_breakpoints=agent_breakpoint, 
            debug_path=debug_path, 
            resume_state={"some": "state"}
        )


@pytest.mark.asyncio
async def test_breakpoint_with_invalid_component_async(mock_agent_with_tool_calls_async, debug_path):
    invalid_bp = Breakpoint(component_name="invalid_breakpoint", visit_count=0)
    with pytest.raises(ValueError, match="All Breakpoints must have component_name 'chat_generator'."):
        AgentBreakpoint({invalid_bp})


@pytest.mark.asyncio
async def test_breakpoint_with_invalid_tool_name_async(mock_agent_with_tool_calls_async, debug_path):
    tool_breakpoint = create_tool_breakpoint(tool_name="invalid_tool", visit_count=0)
    with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
        agent_breakpoints = create_agent_breakpoint(tool_breakpoints={tool_breakpoint})
        await mock_agent_with_tool_calls_async.run_async(
            messages=[ChatMessage.from_user("What's the weather in Berlin?")],
            agent_breakpoints=agent_breakpoints,
            debug_path=debug_path
        )
