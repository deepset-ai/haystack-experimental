# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ChatRole(str, Enum):
    """Enumeration representing the roles within a chat."""

    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ToolCall:
    """
    Represents a Tool call prepared by the model, usually wrapped in an assistant message.

    :param id: The ID of the Tool call.
    :param tool_name: The name of the Tool to call.
    :param arguments: The arguments to call the Tool with.
    """

    id: str  # noqa: A003
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ChatMessage:
    """
    Represents a message in a LLM chat conversation.

    :param content: The text content of the message.
    :param role: The role of the entity sending the message.
    :tool_call_id: The ID of the tool call that this message is responding to. Applies only to messages from tools.
    :tool_calls: List of prepared tool calls. Applies only to messages from the assistant.
    :param meta: Additional metadata associated with the message.
    """

    content: str
    role: ChatRole
    tool_call_id: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)

    def is_from(self, role: ChatRole) -> bool:
        """
        Check if the message is from a specific role.

        :param role: The role to check against.
        :returns: True if the message is from the specified role, False otherwise.
        """
        return self.role == role

    @classmethod
    def from_assistant(
        cls, content: str, tool_calls: Optional[List[ToolCall]] = None, meta: Optional[Dict[str, Any]] = None
    ) -> "ChatMessage":
        """
        Create a message from the assistant.

        :param content: The text content of the message.
        :param tool_calls: List of prepared tool calls.
        :param meta: Additional metadata associated with the message.
        :returns: A new ChatMessage instance.
        """
        return cls(
            content=content, role=ChatRole.ASSISTANT, tool_call_id=None, tool_calls=tool_calls or [], meta=meta or {}
        )

    @classmethod
    def from_user(cls, content: str) -> "ChatMessage":
        """
        Create a message from the user.

        :param content: The text content of the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content=content, role=ChatRole.USER, tool_call_id=None, tool_calls=[], meta={})

    @classmethod
    def from_system(cls, content: str) -> "ChatMessage":
        """
        Create a message from the system.

        :param content: The text content of the message.
        :returns: A new ChatMessage instance.
        """
        return cls(content=content, role=ChatRole.SYSTEM, tool_call_id=None, tool_calls=[], meta={})

    @classmethod
    def from_tool(cls, content: str, tool_call_id: str) -> "ChatMessage":
        """
        Create a message from a tool.

        :param content: Content of the tool message.
        :param tool_call_id: Tool call that this message is responding to.
        :returns: A new ChatMessage instance.
        """
        return cls(content=content, role=ChatRole.TOOL, tool_call_id=tool_call_id, tool_calls=[], meta={})

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ChatMessage into a dictionary.

        :returns:
            Serialized version of the object.
        """
        data = asdict(self)
        data["role"] = self.role.value

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """
        Creates a new ChatMessage object from a dictionary.

        :param data:
            The dictionary to build the ChatMessage object.
        :returns:
            The created object.
        """
        data["role"] = ChatRole(data["role"])

        # deserialize tool_calls
        if tool_calls := data.get("tool_calls"):
            data["tool_calls"] = [ToolCall(**tool_call) for tool_call in tool_calls]

        return cls(**data)
