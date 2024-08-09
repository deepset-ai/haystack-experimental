# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from openai import Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat import chat_completion_chunk

# fixtures copied from Haystack
@pytest.fixture
def mock_chat_completion_chunk():
    """
    Mock the OpenAI API completion chunk response and reuse it for tests
    """

    class MockStream(Stream[ChatCompletionChunk]):
        def __init__(self, mock_chunk: ChatCompletionChunk, client=None, *args, **kwargs):
            client = client or MagicMock()
            super().__init__(client=client, *args, **kwargs)
            self.mock_chunk = mock_chunk

        def __stream__(self) -> Iterator[ChatCompletionChunk]:
            # Yielding only one ChatCompletionChunk object
            yield self.mock_chunk

    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletionChunk(
            id="foo",
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[
                chat_completion_chunk.Choice(
                    finish_reason="stop", logprobs=None, index=0, delta=chat_completion_chunk.ChoiceDelta(content="Hello", role="assistant")
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )
        mock_chat_completion_create.return_value = MockStream(completion, cast_to=None, response=None, client=None)
        yield mock_chat_completion_create

@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="gpt-4",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now().timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create        