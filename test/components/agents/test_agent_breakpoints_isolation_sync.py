# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
from pathlib import Path

from haystack.dataclasses import ChatMessage

from haystack_experimental.core.errors import BreakpointException
from haystack_experimental.core.pipeline.breakpoint import load_state
from haystack_experimental.dataclasses.breakpoints import AgentBreakpoint, Breakpoint

from test.components.agents.test_agent_breakpoints_utils import (
    create_chat_generator_breakpoint,
    create_tool_breakpoint,
    weather_tool,
    debug_path,
    agent_sync,
    mock_agent_with_tool_calls_sync,
)

agent_name = "isolated_agent"

def test_run_with_chat_generator_breakpoint(agent_sync, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    chat_generator_bp = create_chat_generator_breakpoint(visit_count=0)
    agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name="test_agent")
    with pytest.raises(BreakpointException) as exc_info:
        agent_sync.run(messages=messages, break_point=agent_breakpoint, debug_path=debug_path, agent_name="test")
    assert exc_info.value.component == "chat_generator"
    assert "messages" in exc_info.value.state


def test_run_with_tool_invoker_breakpoint(mock_agent_with_tool_calls_sync, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = create_tool_breakpoint(tool_name="weather_tool", visit_count=0)
    agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")
    with pytest.raises(BreakpointException) as exc_info:
        mock_agent_with_tool_calls_sync.run(
            messages=messages,
            break_point=agent_breakpoint,
            debug_path=debug_path,
            agent_name="test"
        )

    assert exc_info.value.component == "tool_invoker"
    assert "messages" in exc_info.value.state


def test_resume_from_chat_generator(agent_sync, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    chat_generator_bp = create_chat_generator_breakpoint(visit_count=0)
    agent_breakpoint = AgentBreakpoint(break_point=chat_generator_bp, agent_name=agent_name)

    try:
        agent_sync.run(messages=messages, break_point=agent_breakpoint, debug_path=debug_path, agent_name=agent_name)
    except BreakpointException:
        pass

    state_files = list(Path(debug_path).glob(agent_name+"_chat_generator_*.json"))

    for f in list(Path(debug_path).glob("*")):
        print(f)

    assert len(state_files) > 0
    latest_state_file = str(max(state_files, key=os.path.getctime))

    resume_state = load_state(latest_state_file)
    result = agent_sync.run(
        messages=[ChatMessage.from_user("Continue from where we left off.")],
        resume_state=resume_state
    )

    assert "messages" in result
    assert "last_message" in result
    assert len(result["messages"]) > 0


def test_resume_from_tool_invoker(mock_agent_with_tool_calls_sync, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = create_tool_breakpoint(tool_name="weather_tool", visit_count=0)
    agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name=agent_name)

    try:
        mock_agent_with_tool_calls_sync.run(
            messages=messages,
            break_point=agent_breakpoint,
            debug_path=debug_path,
            agent_name=agent_name
        )
    except BreakpointException:
        pass

    state_files = list(Path(debug_path).glob(agent_name+"_tool_invoker_*.json"))

    for f in list(Path(debug_path).glob("*")):
        print(f)


    assert len(state_files) > 0
    latest_state_file = str(max(state_files, key=os.path.getctime))

    resume_state = load_state(latest_state_file)

    result = mock_agent_with_tool_calls_sync.run(
        messages=[ChatMessage.from_user("Continue from where we left off.")],
        resume_state=resume_state
    )

    assert "messages" in result
    assert "last_message" in result
    assert len(result["messages"]) > 0


def test_invalid_combination_breakpoint_and_resume_state(mock_agent_with_tool_calls_sync, debug_path):
    messages = [ChatMessage.from_user("What's the weather in Berlin?")]
    tool_bp = create_tool_breakpoint(tool_name="weather_tool", visit_count=0)
    agent_breakpoint = AgentBreakpoint(break_point=tool_bp, agent_name="test_agent")
    with pytest.raises(ValueError, match="agent_breakpoint and resume_state cannot be provided at the same time"):
        mock_agent_with_tool_calls_sync.run(
            messages=messages,
            break_point=agent_breakpoint,
            debug_path=debug_path,
            resume_state={"some": "state"}
        )


def test_breakpoint_with_invalid_component(mock_agent_with_tool_calls_sync, debug_path):
    invalid_bp = Breakpoint(component_name="invalid_breakpoint", visit_count=0)
    with pytest.raises(ValueError):
        AgentBreakpoint(break_point=invalid_bp, agent_name="test_agent")


def test_breakpoint_with_invalid_tool_name(mock_agent_with_tool_calls_sync, debug_path):
    tool_breakpoint = create_tool_breakpoint(tool_name="invalid_tool", visit_count=0)
    with pytest.raises(ValueError, match="Tool 'invalid_tool' is not available in the agent's tools"):
        agent_breakpoints = AgentBreakpoint(break_point=tool_breakpoint, agent_name="test_agent")
        mock_agent_with_tool_calls_sync.run(
            messages=[ChatMessage.from_user("What's the weather in Berlin?")],
            break_point=agent_breakpoints,
            debug_path=debug_path
        )
