# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from haystack import logging
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.dataclasses.breakpoints import AgentBreakpoint, PipelineSnapshot, PipelineState, ToolBreakpoint
from haystack.utils.base_serialization import _serialize_value_with_schema
from haystack.utils.misc import _get_output_dir

from haystack_experimental.dataclasses.breakpoints import AgentSnapshot

if TYPE_CHECKING:
    from haystack_experimental.components.agents.agent import _ExecutionContext
    from haystack_experimental.components.agents.human_in_the_loop import ToolExecutionDecision


logger = logging.getLogger(__name__)


def _create_agent_snapshot(
    *,
    component_visits: dict[str, int],
    agent_breakpoint: AgentBreakpoint,
    component_inputs: dict[str, Any],
    tool_execution_decisions: Optional[list["ToolExecutionDecision"]] = None,
) -> AgentSnapshot:
    """
    Create a snapshot of the agent's state.

    NOTE: Only difference to Haystack's native implementation is the addition of tool_execution_decisions to the
    AgentSnapshot.

    :param component_visits: The visit counts for the agent's components.
    :param agent_breakpoint: AgentBreakpoint object containing breakpoints
    :param component_inputs: The inputs to the agent's components.
    :param tool_execution_decisions: Optional list of ToolExecutionDecision objects representing decisions made
        regarding tool executions.
    :return: An AgentSnapshot containing the agent's state and component visits.
    """
    return AgentSnapshot(
        component_inputs={
            "chat_generator": _serialize_value_with_schema(
                _deepcopy_with_exceptions(component_inputs["chat_generator"])
            ),
            "tool_invoker": _serialize_value_with_schema(_deepcopy_with_exceptions(component_inputs["tool_invoker"])),
        },
        component_visits=component_visits,
        break_point=agent_breakpoint,
        timestamp=datetime.now(),
        tool_execution_decisions=tool_execution_decisions,
    )


def _create_pipeline_snapshot_from_tool_invoker(
    *,
    execution_context: "_ExecutionContext",
    tool_name: Optional[str] = None,
    agent_name: Optional[str] = None,
    break_point: Optional[AgentBreakpoint] = None,
    parent_snapshot: Optional[PipelineSnapshot] = None,
) -> PipelineSnapshot:
    """
    Create a pipeline snapshot when a tool invoker breakpoint is raised or an exception during execution occurs.

    :param execution_context: The current execution context of the agent.
    :param tool_name: The name of the tool that triggered the breakpoint, if available.
    :param agent_name: The name of the agent component if present in a pipeline.
    :param break_point: An optional AgentBreakpoint object. If provided, it will be used instead of creating a new one.
        A scenario where a new breakpoint is created is when an exception occurs during tool execution and we want to
        capture the state at that point.
    :param parent_snapshot: An optional parent PipelineSnapshot to build upon.
    :returns:
        A PipelineSnapshot containing the state of the pipeline and agent at the point of the breakpoint or exception.
    """
    if break_point is None:
        agent_breakpoint = AgentBreakpoint(
            agent_name=agent_name or "agent",
            break_point=ToolBreakpoint(
                component_name="tool_invoker",
                visit_count=execution_context.component_visits["tool_invoker"],
                tool_name=tool_name,
                snapshot_file_path=_get_output_dir("pipeline_snapshot"),
            ),
        )
    else:
        agent_breakpoint = break_point

    messages = execution_context.state.data["messages"]
    agent_snapshot = _create_agent_snapshot(
        component_visits=execution_context.component_visits,
        agent_breakpoint=agent_breakpoint,
        component_inputs={
            "chat_generator": {"messages": messages[:-1], **execution_context.chat_generator_inputs},
            "tool_invoker": {
                "messages": messages[-1:],  # tool invoker consumes last msg from the chat_generator, contains tool call
                "state": execution_context.state,
                **execution_context.tool_invoker_inputs,
            },
        },
        tool_execution_decisions=execution_context.tool_execution_decisions,
    )
    if parent_snapshot is None:
        # Create an empty pipeline snapshot if no parent snapshot is provided
        final_snapshot = PipelineSnapshot(
            pipeline_state=PipelineState(inputs={}, component_visits={}, pipeline_outputs={}),
            timestamp=agent_snapshot.timestamp,
            break_point=agent_snapshot.break_point,
            agent_snapshot=agent_snapshot,
            original_input_data={},
            ordered_component_names=[],
            include_outputs_from=set(),
        )
    else:
        final_snapshot = replace(parent_snapshot, agent_snapshot=agent_snapshot)

    return final_snapshot
