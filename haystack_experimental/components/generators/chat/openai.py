# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import haystack.components.generators.chat.openai
from haystack_experimental.dataclasses.chat_message import ChatMessage
from haystack import component

# Monkey patch the Haystack ChatMessage class with the experimental one. By doing so, we can use the new 
# `to_openai_dict_format` method, allowing multimodal chat messages.
haystack.components.generators.chat.openai.ChatMessage = ChatMessage

@component
class OpenAIChatGenerator(haystack.components.generators.chat.openai.OpenAIChatGenerator):
    """
    Experimental version of OpenAIChatGenerator that allows multimodal chat messages.
    """
    pass
        
        
