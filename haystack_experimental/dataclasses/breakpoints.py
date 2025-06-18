# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional, Set, Union


@dataclass
class Breakpoint:
    """
    A dataclass to hold a breakpoint for a component.
    """

    component_name: str
    visit_count: int

    def __hash__(self):
        return hash((self.component_name, self.visit_count))

    def __eq__(self, other):
        if not isinstance(other, Breakpoint):
            return False
        return self.component_name == other.component_name and self.visit_count == other.visit_count

    def __str__(self):
        return f"Breakpoint(component_name={self.component_name}, visit_count={self.visit_count})"

    def __repr__(self):
        return self.__str__()


@dataclass
class ToolBreakpoint(Breakpoint):
    """
    A dataclass to hold a breakpoint that can be used to debug a Tool.

    If tool_name is None, it means that the breakpoint is for every tool in the component.
    Otherwise, it means that the breakpoint is for the tool with the given name.
    """

    tool_name: Optional[str] = None

    def __hash__(self):
        return hash((self.component_name, self.visit_count, self.tool_name))

    def __eq__(self, other):
        if not isinstance(other, ToolBreakpoint):
            return False
        return super().__eq__(other)

    def __str__(self):
        if self.tool_name:
            return (
                f"ToolBreakpoint(component_name={self.component_name}, visit_count={self.visit_count}, "
                f"tool_name={self.tool_name})"
            )
        else:
            return (
                f"ToolBreakpoint(component_name={self.component_name}, visit_count={self.visit_count}, "
                f"tool_name=ALL_TOOLS)"
            )

    def __repr__(self):
        return self.__str__()


@dataclass
class AgentBreakpoint:
    """
    A dataclass to hold a breakpoint that can be used to debug an Agent.

    It holds a set of breakpoints for the components in the Agent.
    """

    breakpoints: Set[Union[Breakpoint, ToolBreakpoint]]

    def __init__(self, breakpoints: Set[Union[Breakpoint, ToolBreakpoint]]):
        if breakpoints is None:
            breakpoints = set()
        self.breakpoints = breakpoints

    def add_breakpoint(self, break_point: Union[Breakpoint, ToolBreakpoint]):
        """
        Adds a breakpoint to the set of breakpoints.
        """
        self.breakpoints.add(break_point)

    def remove_breakpoint(self, break_point: Union[Breakpoint, ToolBreakpoint]):
        """
        Removes a breakpoint from the set of breakpoints.
        """
        self.breakpoints.remove(break_point)
