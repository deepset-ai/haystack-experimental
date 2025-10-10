# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.dataclasses.breakpoints import AgentBreakpoint, ToolBreakpoint

from haystack_experimental.dataclasses.breakpoints import AgentSnapshot
from haystack_experimental.components.agents.human_in_the_loop.breakpoint import (
    get_tool_calls_and_descriptions_from_snapshot,
)


def get_bank_balance(account_id: str) -> str:
    return f"The balance for account {account_id} is $1,234.56."


def addition(a: float, b: float) -> float:
    return a + b


def test_get_tool_calls_and_descriptions_from_snapshot():
    agent_snapshot = AgentSnapshot(
        component_inputs={
            "chat_generator": {},
            "tool_invoker": {
                "serialization_schema": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"},
                        },
                        "state": {"type": "haystack.components.agents.state.state.State"},
                        "tools": {"type": "array", "items": {"type": "haystack.tools.tool.Tool"}},
                        "enable_streaming_callback_passthrough": {"type": "boolean"},
                    },
                },
                "serialized_data": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "tool_call": {
                                        "tool_name": "get_bank_balance",
                                        "arguments": {"account_id": "56789"},
                                        "id": None,
                                    }
                                }
                            ],
                        }
                    ],
                    "state": {
                        "schema": {
                            "messages": {
                                "type": "list[haystack.dataclasses.chat_message.ChatMessage]",
                                "handler": "haystack.components.agents.state.state_utils.merge_lists",
                            }
                        },
                        "data": {
                            "serialization_schema": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {"type": "haystack.dataclasses.chat_message.ChatMessage"},
                                    }
                                },
                            },
                            "serialized_data": {
                                "messages": [
                                    {
                                        "role": "system",
                                        "content": [
                                            {
                                                "text": "You are a helpful financial assistant. Use the provided tool to get bank balances when needed."
                                            }
                                        ],
                                    },
                                    {
                                        "role": "user",
                                        "content": [{"text": "What's the balance of account 56789?"}],
                                    },
                                    {
                                        "role": "assistant",
                                        "content": [
                                            {
                                                "tool_call": {
                                                    "tool_name": "get_bank_balance",
                                                    "arguments": {"account_id": "56789"},
                                                    "id": None,
                                                }
                                            }
                                        ],
                                    },
                                ]
                            },
                        },
                    },
                    "tools": [
                        {
                            "type": "haystack.tools.tool.Tool",
                            "data": {
                                "name": "get_bank_balance",
                                "description": "Get the bank balance for a given account ID.",
                                "parameters": {
                                    "properties": {"account_id": {"type": "string"}},
                                    "required": ["account_id"],
                                    "type": "object",
                                },
                                "function": "test.components.agents.human_in_the_loop.test_breakpoint.get_bank_balance",
                            },
                        },
                        {
                            "type": "haystack.tools.tool.Tool",
                            "data": {
                                "name": "addition",
                                "description": "Add two floats together.",
                                "parameters": {
                                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                                    "required": ["a", "b"],
                                    "type": "object",
                                },
                                "function": "test.components.agents.human_in_the_loop.test_breakpoint.addition",
                            },
                        },
                    ],
                    "enable_streaming_callback_passthrough": False,
                },
            },
        },
        component_visits={"chat_generator": 1, "tool_invoker": 0},
        break_point=AgentBreakpoint(
            agent_name="agent",
            break_point=ToolBreakpoint(
                tool_name="get_bank_balance", component_name="tool_invoker", visit_count=0, snapshot_file_path=None
            ),
        ),
    )

    tool_calls, tool_descriptions = get_tool_calls_and_descriptions_from_snapshot(
        agent_snapshot=agent_snapshot, breakpoint_tool_only=True
    )

    assert len(tool_calls) == 1
    assert tool_calls[0]["tool_name"] == "get_bank_balance"
    assert tool_calls[0]["arguments"] == {"account_id": "56789"}
    assert tool_descriptions == {"get_bank_balance": "Get the bank balance for a given account ID."}
