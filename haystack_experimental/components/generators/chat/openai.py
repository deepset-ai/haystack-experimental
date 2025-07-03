# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import haystack.components.generators.chat.openai
from haystack import component

from haystack_experimental.dataclasses.chat_message import ChatMessage

# Monkey patch the Haystack ChatMessage class with the experimental one. By doing so, we can use the new
# `to_openai_dict_format` method, allowing multimodal chat messages.
haystack.components.generators.chat.openai.ChatMessage = ChatMessage  # type: ignore[misc]


@component
class OpenAIChatGenerator(haystack.components.generators.chat.openai.OpenAIChatGenerator):
    """
    Experimental version of OpenAIChatGenerator that allows multimodal chat messages.

    ### Usage example
    ```python
    from haystack_experimental.components.generators.chat import OpenAIChatGenerator
    from haystack_experimental.dataclasses import ChatMessage, ImageContent

    generator = OpenAIChatGenerator(model="gpt-4o-mini")

    image_content = ImageContent.from_file_path(file_path="apple.jpg")

    message = ChatMessage.from_user(content_parts=["Please describe the image using 5 words at most.", image_content])

    response = generator.run(messages=[message])["replies"][0].text

    print(response)
    # Red apple on straw background.
    ```
    """

    pass
