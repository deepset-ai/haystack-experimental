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
    visit_count: int = 0

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
        return super().__eq__(other) and self.tool_name == other.tool_name

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

    generator_breakpoints: Set[Breakpoint]
    tool_breakpoints: Set[ToolBreakpoint]

    def __init__(self, breakpoints: Set[Union[Breakpoint, ToolBreakpoint]]):
        for break_point in breakpoints:
            if not isinstance(break_point, ToolBreakpoint) and break_point.component_name != "chat_generator":
                raise ValueError("All Breakpoints must have component_name 'chat_generator'.")

        if not breakpoints:
            raise ValueError("Breakpoints must be provided.")

        self.tool_breakpoints = set()
        self.generator_breakpoints = set()

        for break_point in breakpoints:
            if isinstance(break_point, ToolBreakpoint):
                self.tool_breakpoints.add(break_point)
            elif isinstance(break_point, Breakpoint) and not isinstance(break_point, ToolBreakpoint):
                self.generator_breakpoints.add(break_point)
            else:
                raise ValueError("Breakpoints must be either Breakpoint or ToolBreakpoint.")

    def add_breakpoint(self, break_point: Union[Breakpoint, ToolBreakpoint]) -> None:
        """
        Adds a breakpoint to the set of breakpoints.
        """

        if isinstance(break_point, Breakpoint):
            if break_point in self.generator_breakpoints:
                raise ValueError(f"Breakpoint {break_point} already exists in generator breakpoints.")
            self.generator_breakpoints.add(break_point)

        if isinstance(break_point, ToolBreakpoint):
            if break_point in self.tool_breakpoints:
                raise ValueError(f"Breakpoint {break_point} already exists in tool breakpoints.")
            self.tool_breakpoints.add(break_point)

    def remove_breakpoint(self, break_point: Union[Breakpoint, ToolBreakpoint]) -> None:
        """
        Removes a breakpoint from the set of breakpoints.
        """

        if isinstance(break_point, Breakpoint):
            if break_point not in self.generator_breakpoints:
                raise ValueError(f"Breakpoint {break_point} does not exist in generator breakpoints.")
            self.generator_breakpoints.remove(break_point)

        if isinstance(break_point, ToolBreakpoint):
            if break_point not in self.tool_breakpoints:
                raise ValueError(f"Breakpoint {break_point} does not exist in tool breakpoints.")
            self.tool_breakpoints.remove(break_point)
