# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.writers import ChatMessageWriter
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore


@pytest.fixture
def store():
    msg_store = InMemoryChatMessageStore()
    yield msg_store
    msg_store.delete_all_messages()


class TestChatMessageWriter:
    def test_init(self, store):
        """
        Test that the ChatMessageWriter component can be initialized with a valid message store.
        """
        messages = [
            ChatMessage.from_user("Hello, how can I help you?"),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?"),
        ]
        writer = ChatMessageWriter(store)
        assert writer.chat_message_store == store
        assert writer.run(chat_history_id="test", messages=messages) == {"messages_written": 2}

    def test_to_dict(self, store):
        """
        Test that the ChatMessageWriter component can be serialized to a dictionary.
        """
        writer = ChatMessageWriter(store)
        data = writer.to_dict()
        assert data == {
            "type": "haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter",
            "init_parameters": {
                "chat_message_store": {
                    "init_parameters": {
                        "skip_system_messages": True,
                        "last_k": 10
                    },
                    "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore",
                }
            },
        }

        # write again and serialize
        writer.run(chat_history_id="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        data = writer.to_dict()
        assert data == {
            "type": "haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter",
            "init_parameters": {
                "chat_message_store": {
                    "init_parameters": {
                        "skip_system_messages": True,
                        "last_k": 10
                    },
                    "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore",
                }
            },
        }

    def test_from_dict(self):
        """
        Test that the ChatMessageWriter component can be deserialized from a dictionary.
        """
        data = {
            "type": "haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter",
            "init_parameters": {
                "chat_message_store": {
                    "init_parameters": {
                        "skip_system_messages": True,
                        "last_k": 10
                    },
                    "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore",
                }
            },
        }
        writer = ChatMessageWriter.from_dict(data)
        assert writer.chat_message_store.to_dict() == {
            "init_parameters": {
                "skip_system_messages": True,
                "last_k": 10
            },
            "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore",
        }

        # write to verify that everything is still working
        results = writer.run(chat_history_id="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        assert results["messages_written"] == 1

        # Cleanup
        writer.chat_message_store.delete_messages(chat_history_id="test")

    def test_chat_message_writer_pipeline(self, store):
        """
        Test that the ChatMessageWriter can be used in a pipeline and that it works as expected.
        """
        user_prompt = """
        Given the following information, answer the question.
        Question: {{ query }}
        Answer:
        """

        pipe = Pipeline()
        pipe.add_component("prompt_builder", ChatPromptBuilder(
            template=[ChatMessage.from_user(user_prompt)], variables=["query"])
        )
        pipe.add_component("writer", ChatMessageWriter(store))
        pipe.connect("prompt_builder", "writer")

        question = "What is the capital of France?"

        res = pipe.run(data={"prompt_builder": {"query": question}, "writer": {"chat_history_id": "test"}})
        assert res["writer"]["messages_written"] == 1  # only one message is written
        assert len(store.retrieve_messages(chat_history_id="test")) == 1  # only one message is written
        assert (
            store.retrieve_messages(chat_history_id="test")[0].text
            == """
        Given the following information, answer the question.
        Question: What is the capital of France?
        Answer:
        """
        )

    def test_chat_message_writer_pipeline_serde(self, store):
        """
        Test that the ChatMessageWriter can be serialized and deserialized.
        """
        pipe = Pipeline()
        pipe.add_component("writer", ChatMessageWriter(store))
        pipe.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("no template")], variables=["query"])
        )

        # now serialize and deserialize the pipeline
        data = pipe.to_dict()
        new_pipe = Pipeline.from_dict(data)

        assert new_pipe == pipe
