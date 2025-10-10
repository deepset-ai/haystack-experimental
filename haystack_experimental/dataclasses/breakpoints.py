# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from haystack.dataclasses.breakpoints import AgentBreakpoint
from haystack.dataclasses.breakpoints import AgentSnapshot as HaystackAgentSnapshot

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ToolExecutionDecision


@dataclass
class AgentSnapshot(HaystackAgentSnapshot):
    tool_execution_decisions: Optional[list[ToolExecutionDecision]] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the AgentSnapshot to a dictionary representation.

        :return: A dictionary containing the agent state, timestamp, and breakpoint.
        """
        return {
            "component_inputs": self.component_inputs,
            "component_visits": self.component_visits,
            "break_point": self.break_point.to_dict(),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tool_execution_decisions": [ted.to_dict() for ted in self.tool_execution_decisions]
            if self.tool_execution_decisions
            else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentSnapshot":
        """
        Populate the AgentSnapshot from a dictionary representation.

        :param data: A dictionary containing the agent state, timestamp, and breakpoint.
        :return: An instance of AgentSnapshot.
        """
        return cls(
            component_inputs=data["component_inputs"],
            component_visits=data["component_visits"],
            break_point=AgentBreakpoint.from_dict(data["break_point"]),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            tool_execution_decisions=[
                ToolExecutionDecision.from_dict(ted) for ted in data.get("tool_execution_decisions", [])
            ]
            if data.get("tool_execution_decisions")
            else None,
        )
