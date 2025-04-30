# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import filetype
import haystack.dataclasses.chat_message
from haystack import logging
from haystack.dataclasses import ChatMessage as HaystackChatMessage
from haystack.dataclasses import ChatRole, TextContent, ToolCall, ToolCallResult
import json

logger = logging.getLogger(__name__)


@dataclass
class ImageContent:
    """
    The image content of a chat message.

    :param base64_image: A base64 string representing the image.
    :param mime_type: The MIME type of the image (e.g. "image/png", "image/jpeg").
        Providing this value is recommended, as most LLM providers require it.
        If not provided, the MIME type is guessed from the base64 string, which can be slow and not always reliable.
    :param detail: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
    :param meta: Optional metadata for the image.
    """

    base64_image: str
    mime_type: Optional[str] = None
    detail: Optional[Literal["auto", "high", "low"]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # mime_type is an important information, so we try to guess it if not provided
        if not self.mime_type:
            try:
                # Attempt to decode the string as base64
                decoded_image = base64.b64decode(self.base64_image)

                guess = filetype.guess(decoded_image)
                if guess:
                    self.mime_type = guess.mime
                else:
                    msg = (
                        "Failed to guess the MIME type of the image. Omitting the MIME type may result in "
                        "processing errors or incorrect handling of the image by LLM providers."
                    )
                    logger.warning(msg)
            except:
                pass

    def __repr__(self) -> str:
        """
        Return a string representation of the ImageContent, truncating the base64_image to 100 bytes.
        """
        fields = []

        truncated_data = self.base64_image[:100] + "..." if len(self.base64_image) > 100 else self.base64_image
        fields.append(f"base64_image={truncated_data!r}")
        fields.append(f"mime_type={self.mime_type!r}")
        fields.append(f"detail={self.detail!r}")
        fields.append(f"meta={self.meta!r}")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}({fields_str})"


ChatMessageContentT = Union[TextContent, ToolCall, ToolCallResult, ImageContent]


def _deserialize_content(serialized_content: List[Dict[str, Any]]) -> List[ChatMessageContentT]:
    """
    Deserialize the `content` field of a serialized ChatMessage.

    :param serialized_content:
        The `content` field of a serialized ChatMessage (a list of dictionaries).

    :returns:
        Deserialized `content` field as a list of `ChatMessageContentT` objects.
    """
    content: List[ChatMessageContentT] = []

    for part in serialized_content:
        if "text" in part:
            content.append(TextContent(text=part["text"]))
        elif "tool_call" in part:
            content.append(ToolCall(**part["tool_call"]))
        elif "tool_call_result" in part:
            result = part["tool_call_result"]["result"]
            origin = ToolCall(**part["tool_call_result"]["origin"])
            error = part["tool_call_result"]["error"]
            tcr = ToolCallResult(result=result, origin=origin, error=error)
            content.append(tcr)
        elif "image" in part:
            content.append(ImageContent(**part["image"]))
        else:
            raise ValueError(f"Unsupported part in serialized ChatMessage: `{part}`")

    return content


# Note: this is a monkey patch to the original _deserialize_content function
haystack.dataclasses.chat_message._deserialize_content = _deserialize_content


@dataclass
class ChatMessage(HaystackChatMessage):
    """
    Represents a message in a LLM chat conversation.

    Use the `from_assistant`, `from_user`, `from_system`, and `from_tool` class methods to create a ChatMessage.
    """

    _role: ChatRole
    _content: Sequence[ChatMessageContentT]
    _name: Optional[str] = None
    _meta: Dict[str, Any] = field(default_factory=dict, hash=False)

    @property
    def images(self) -> List[ImageContent]:
        """
        Returns the list of all images contained in the message.
        """
        return [content for content in self._content if isinstance(content, ImageContent)]

    @property
    def image(self) -> Optional[ImageContent]:
        """
        Returns the first image contained in the message.
        """
        if images := self.images:
            return images[0]
        return None

    @classmethod
    def from_user(
        cls,
        text: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        *,
        content_parts: Optional[Sequence[Union[TextContent, str, ImageContent]]] = None,
    ) -> "ChatMessage":
        """
        Create a message from the user.

        :param text: The text content of the message. Specify this or content_parts.
        :param meta: Additional metadata associated with the message.
        :param name: An optional name for the participant. This field is only supported by OpenAI.
        :param content_parts: A list of content parts to include in the message. Specify this or text.
        :returns: A new ChatMessage instance.
        """
        if not text and not content_parts:
            raise ValueError("Either text or content_parts must be provided.")
        if text and content_parts:
            raise ValueError("Only one of text or content_parts can be provided.")

        content: Sequence[Union[TextContent, ImageContent]] = []

        if text is not None:
            content = [TextContent(text=text)]
        elif content_parts is not None:
            content = [TextContent(el) if isinstance(el, str) else el for el in content_parts]
            if not any(isinstance(el, TextContent) for el in content):
                raise ValueError("The user message must contain at least one textual part.")

        return cls(_role=ChatRole.USER, _content=content, _meta=meta or {}, _name=name)

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts ChatMessage into a dictionary.

        :returns:
            Serialized version of the object.
        """

        serialized: Dict[str, Any] = {}
        serialized["role"] = self._role.value
        serialized["meta"] = self._meta
        serialized["name"] = self._name
        content: List[Dict[str, Any]] = []
        for part in self._content:
            if isinstance(part, TextContent):
                content.append({"text": part.text})
            elif isinstance(part, ToolCall):
                content.append({"tool_call": asdict(part)})
            elif isinstance(part, ToolCallResult):
                content.append({"tool_call_result": asdict(part)})
            elif isinstance(part, ImageContent):
                content.append({"image": asdict(part)})
            else:
                raise TypeError(f"Unsupported type in ChatMessage content: `{type(part).__name__}` for `{part}`.")

        serialized["content"] = content
        return serialized
    
    def to_openai_dict_format(self) -> Dict[str, Any]:
        """
        Convert a ChatMessage to the dictionary format expected by OpenAI's Chat API.
        """
        text_contents = self.texts
        tool_calls = self.tool_calls
        tool_call_results = self.tool_call_results

        if not text_contents and not tool_calls and not tool_call_results:
            raise ValueError(
                "A `ChatMessage` must contain at least one `TextContent`, `ToolCall`, or `ToolCallResult`."
            )
        if len(text_contents) + len(tool_call_results) > 1:
            raise ValueError("A `ChatMessage` can only contain one `TextContent` or one `ToolCallResult`.")

        openai_msg: Dict[str, Any] = {"role": self._role.value}

        # Add name field if present
        if self._name is not None:
            openai_msg["name"] = self._name

        # user message
        if openai_msg["role"] == "user":
            if len(self._content) == 1:
                openai_msg["content"] = self.text
                return openai_msg

            # if the user message contains a list of text and images, OpenAI expects a list of dictionaries
            content = []
            for part in self._content:
                if isinstance(part, TextContent):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContent):
                    image_item: Dict[str, Any] = {
                        "type": "image_url",
                        # If no MIME type is provided, default to JPEG.
                        # OpenAI API appears to tolerate MIME type mismatches.
                        "image_url": {"url": f"data:{part.mime_type or 'image/jpeg'};base64,{part.base64_image}"},
                    }
                    if part.detail:
                        image_item["image_url"]["detail"] = part.detail
                    content.append(image_item)
            openai_msg["content"] = content
            return openai_msg            

        # tool message
        if tool_call_results:
            result = tool_call_results[0]
            if result.origin.id is None:
                raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
            openai_msg["content"] = result.result
            openai_msg["tool_call_id"] = result.origin.id
            # OpenAI does not provide a way to communicate errors in tool invocations, so we ignore the error field
            return openai_msg

        # system and assistant messages
        if text_contents:
            openai_msg["content"] = text_contents[0]
        if tool_calls:
            openai_tool_calls = []
            for tc in tool_calls:
                if tc.id is None:
                    raise ValueError("`ToolCall` must have a non-null `id` attribute to be used with OpenAI.")
                openai_tool_calls.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        # We disable ensure_ascii so special chars like emojis are not converted
                        "function": {"name": tc.tool_name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                )
            openai_msg["tool_calls"] = openai_tool_calls
        return openai_msg
