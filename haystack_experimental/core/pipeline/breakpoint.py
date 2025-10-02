# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from haystack import logging
from haystack.dataclasses.breakpoints import AgentBreakpoint
from haystack.utils.base_serialization import _serialize_value_with_schema

from haystack_experimental.dataclasses.breakpoints import AgentSnapshot

if TYPE_CHECKING:
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
            "chat_generator": _serialize_value_with_schema(deepcopy(component_inputs["chat_generator"])),
            "tool_invoker": _serialize_value_with_schema(deepcopy(component_inputs["tool_invoker"])),
        },
        component_visits=component_visits,
        break_point=agent_breakpoint,
        timestamp=datetime.now(),
        tool_execution_decisions=tool_execution_decisions,
    )
