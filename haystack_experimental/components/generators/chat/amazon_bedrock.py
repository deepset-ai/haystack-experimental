# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from typing import Any, Dict, List, Optional, Tuple, Union

from haystack import component
from haystack.dataclasses.streaming_chunk import StreamingCallbackT
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret

from haystack_experimental.dataclasses.chat_message import ChatMessage, ChatRole, ImageContent, TextContent

with LazyImport("Run 'pip install amazon-bedrock-haystack'") as bedrock_integration_import:
    import haystack_integrations.components.generators.amazon_bedrock.chat.chat_generator as original_chat_generator
    import haystack_integrations.components.generators.amazon_bedrock.chat.utils as original_utils
    from haystack_integrations.components.generators.amazon_bedrock.chat.utils import (
        _format_tool_call_message,
        _format_tool_result_message,
        _repair_tool_result_messages,
    )

# NOTE: The following implementation ensures that:
# - we reuse existing code where possible
# - people can use haystack-experimental without installing amazon-bedrock-haystack.
#
#
# If amazon-bedrock-haystack is installed: all works correctly.
#
# If amazon-bedrock-haystack is not installed:
# - haystack-experimental package works fine (no import errors).
# - AmazonBedrockChatGenerator fails with ImportError at init (due to bedrock_integration_import.check()).

if not bedrock_integration_import.is_successful():

    @component
    class AmazonBedrockChatGenerator:
        """
        Experimental version of AmazonBedrockChatGenerator that allows multimodal chat messages.

        ### Usage example
        ```python
        from haystack_experimental.components.generators.chat import AmazonBedrockChatGenerator
        from haystack_experimental.dataclasses import ChatMessage, ImageContent

        generator = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")

        image_content = ImageContent.from_file_path(file_path="apple.jpg")

        message = ChatMessage.from_user(content_parts=["Describe the image using 10 words at most.", image_content])

        response = generator.run(messages=[message])["replies"][0].text

        print(response)
        # The image shows a red apple.
        ```
        """

        def __init__(  # pylint: disable=too-many-positional-arguments
            self,
            model: str,
            aws_access_key_id: Optional[Secret] = Secret.from_env_var(["AWS_ACCESS_KEY_ID"], strict=False),  # noqa: B008
            aws_secret_access_key: Optional[Secret] = Secret.from_env_var(  # noqa: B008
                ["AWS_SECRET_ACCESS_KEY"], strict=False
            ),
            aws_session_token: Optional[Secret] = Secret.from_env_var(["AWS_SESSION_TOKEN"], strict=False),  # noqa: B008
            aws_region_name: Optional[Secret] = Secret.from_env_var(["AWS_DEFAULT_REGION"], strict=False),  # noqa: B008
            aws_profile_name: Optional[Secret] = Secret.from_env_var(["AWS_PROFILE"], strict=False),  # noqa: B008
            generation_kwargs: Optional[Dict[str, Any]] = None,
            stop_words: Optional[List[str]] = None,
            streaming_callback: Optional[StreamingCallbackT] = None,
            boto3_config: Optional[Dict[str, Any]] = None,
            tools: Optional[Union[List[Tool], Toolset]] = None,
        ) -> None:
            bedrock_integration_import.check()  # this always fails

        @component.output_types(replies=List[ChatMessage])
        def run(
            self,
            messages: List[ChatMessage],
            streaming_callback: Optional[StreamingCallbackT] = None,
            generation_kwargs: Optional[Dict[str, Any]] = None,
            tools: Optional[Union[List[Tool], Toolset]] = None,
        ) -> Dict[str, List[ChatMessage]]:
            """
            Executes a synchronous inference call to the Amazon Bedrock model using the Converse API.
            """

            # NOTE: placeholder run method needed to make component happy
            raise NotImplementedError("Unreachable code")
else:
    # see https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html for supported formats
    IMAGE_SUPPORTED_FORMATS = ["png", "jpeg", "gif", "webp"]

    # NOTE: this is the new function needed to support images
    def _format_text_image_message(message: ChatMessage) -> Dict[str, Any]:
        """
        Format a Haystack ChatMessage containing text and optional image content into Bedrock format.

        :param message: Haystack ChatMessage.
        :returns: Dictionary representing the message in Bedrock's expected format.
        :raises ValueError: If image content is found in an assistant message or an unsupported image format is used.
        """
        content_parts = message._content

        bedrock_content_blocks: List[Dict[str, Any]] = []
        for part in content_parts:
            if isinstance(part, TextContent):
                bedrock_content_blocks.append({"text": part.text})

            elif isinstance(part, ImageContent):
                if message.is_from(ChatRole.ASSISTANT):
                    err_msg = "Image content is not supported for assistant messages"
                    raise ValueError(err_msg)

                image_format = part.mime_type.split("/")[-1] if part.mime_type else None
                if image_format not in IMAGE_SUPPORTED_FORMATS:
                    err_msg = (
                        f"Unsupported image format: {image_format}. "
                        f"Bedrock supports the following image formats: {IMAGE_SUPPORTED_FORMATS}"
                    )
                    raise ValueError(err_msg)
                source = {"bytes": base64.b64decode(part.base64_image)}
                bedrock_content_blocks.append({"image": {"format": image_format, "source": source}})

        return {"role": message.role.value, "content": bedrock_content_blocks}

    # NOTE: this is reimplemented in order to call the new _format_text_image_message function
    def _format_messages(messages: List[ChatMessage]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Format a list of Haystack ChatMessages to the format expected by Bedrock API.

        Processes and separates system messages from other message types and handles special formatting for tool calls
        and tool results.

        :param messages: List of ChatMessage objects to format for Bedrock API.
        :returns: Tuple containing (system_prompts, non_system_messages) in Bedrock format,
                where system_prompts is a list of system message dictionaries and
                non_system_messages is a list of properly formatted message dictionaries.
        """
        # Separate system messages, tool calls, and tool results
        system_prompts = []
        bedrock_formatted_messages = []
        for msg in messages:
            if msg.is_from(ChatRole.SYSTEM):
                # Assuming system messages can only contain text
                # Don't need to track idx since system_messages are handled separately
                system_prompts.append({"text": msg.text})
            elif msg.tool_calls:
                bedrock_formatted_messages.append(_format_tool_call_message(msg))
            elif msg.tool_call_results:
                bedrock_formatted_messages.append(_format_tool_result_message(msg))
            else:
                bedrock_formatted_messages.append(_format_text_image_message(msg))

        repaired_bedrock_formatted_messages = _repair_tool_result_messages(bedrock_formatted_messages)
        return system_prompts, repaired_bedrock_formatted_messages

    # NOTE: monkey patches needed to use the new ChatMessage dataclass and _format_messages function
    original_utils.ChatMessage = ChatMessage  # type: ignore[misc]
    original_chat_generator.ChatMessage = ChatMessage  # type: ignore[misc]
    original_chat_generator._format_messages = _format_messages  # type: ignore[assignment]

    @component
    class AmazonBedrockChatGenerator(original_chat_generator.AmazonBedrockChatGenerator):  # type: ignore[no-redef]
        pass
