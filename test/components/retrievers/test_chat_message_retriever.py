import pytest
from typing import Any

from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.writers import ChatMessageWriter


@component
class MockChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant("This is a mock response.")]}


@component
class MockAgent:
    @component.output_types(messages=list[ChatMessage], last_message=ChatMessage)
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        assistant_msg = ChatMessage.from_assistant("This is a mock response.")
        return {"messages": [*messages, assistant_msg], "last_message": assistant_msg}


class TestChatMessageRetriever:
    def test_init(self):
        """
        Test that the ChatMessageRetriever component can be initialized with a valid message store.
        """
        message_store = InMemoryChatMessageStore()
        retriever = ChatMessageRetriever(message_store)

        assert retriever.message_store == message_store
        assert retriever.run(index="test") == {"messages": []}

    def test_retrieve_messages(self):
        """
        Test that the ChatMessageRetriever component can retrieve messages from the message store.
        """
        messages = [
            ChatMessage.from_user("Hello, how can I help you?"),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")
        ]

        message_store = InMemoryChatMessageStore()
        message_store.write_messages(index="test", messages=messages)
        retriever = ChatMessageRetriever(message_store)

        assert retriever.message_store == message_store
        assert retriever.run(index="test") == {"messages": messages}
        # Clean up
        message_store.delete_messages(index="test")

    def test_retrieve_messages_last_k(self):
        """
        Test that the ChatMessageRetriever component can retrieve last_k messages from the message store.
        """
        messages = [
            ChatMessage.from_user("Hello, how can I help you?"),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?"),
            ChatMessage.from_user("Hola, como puedo ayudarte?"),
            ChatMessage.from_user("Bonjour, comment puis-je vous aider?")
        ]

        message_store = InMemoryChatMessageStore()
        message_store.write_messages(index="test", messages=messages)
        retriever = ChatMessageRetriever(message_store)

        assert retriever.message_store == message_store
        assert retriever.run(index="test", last_k=1) == {
            "messages": [ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"is_stored": True})]
        }

        assert retriever.run(index="test", last_k=2) == {
            "messages": [
                ChatMessage.from_user("Hola, como puedo ayudarte?", meta={"is_stored": True}),
                ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"is_stored": True}),
            ]
        }

        # outliers
        assert retriever.run(index="test", last_k=10) == {
            "messages": [
                ChatMessage.from_user("Hello, how can I help you?", meta={"is_stored": True}),
                ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?", meta={"is_stored": True}),
                ChatMessage.from_user("Hola, como puedo ayudarte?", meta={"is_stored": True}),
                ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"is_stored": True}),
            ]
        }

        with pytest.raises(ValueError):
            retriever.run(index="test", last_k=0)

        with pytest.raises(ValueError):
            retriever.run(index="test", last_k=-1)

        # Clean up
        message_store.delete_messages(index="test")

    def test_retrieve_messages_last_k_init(self):
        """
        Test that the ChatMessageRetriever component can retrieve last_k messages from the message store
        by testing the init last_k parameter and the run last_k parameter logic
        """
        messages = [
            ChatMessage.from_user("Hello, how can I help you?"),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?"),
            ChatMessage.from_user("Hola, como puedo ayudarte?"),
            ChatMessage.from_user("Bonjour, comment puis-je vous aider?")
        ]

        message_store = InMemoryChatMessageStore()
        message_store.write_messages(index="test", messages=messages)
        retriever = ChatMessageRetriever(message_store, last_k=2)

        assert retriever.message_store == message_store

        # last_k is 1 here from run parameter, overrides init of 2
        assert retriever.run(index="test", last_k=1) == {
            "messages": [ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"is_stored": True})]
        }

        # last_k is 2 here from init
        assert retriever.run(index="test") == {
            "messages": [
                ChatMessage.from_user("Hola, como puedo ayudarte?", meta={"is_stored": True}),
                ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"is_stored": True}),
            ]
        }

        # Clean up
        message_store.delete_messages(index="test")

    def test_to_dict(self):
        """
        Test that the ChatMessageRetriever component can be serialized to a dictionary.
        """
        message_store = InMemoryChatMessageStore()
        retriever = ChatMessageRetriever(message_store, last_k=4)

        data = retriever.to_dict()
        assert data == {
            "type": "haystack_experimental.components.retrievers.chat_message_retriever.ChatMessageRetriever",
            "init_parameters": {
                "message_store": {
                    "init_parameters": {},
                    "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"
                },
                "last_k": 4,
            },
        }

    def test_from_dict(self):
        """
        Test that the ChatMessageRetriever component can be deserialized from a dictionary.
        """
        data = {
            "type": "haystack_experimental.components.retrievers.chat_message_retriever.ChatMessageRetriever",
            "init_parameters": {
                "message_store": {
                    "init_parameters": {},
                    "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"
                },
                "last_k": 4,
            },
        }
        retriever = ChatMessageRetriever.from_dict(data)
        assert retriever.message_store.to_dict() == {
            "init_parameters": {},
            "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"
        }
        assert retriever.last_k == 4

    def test_chat_message_retriever_pipeline(self):
        index = "user_123_session_456"
        store = InMemoryChatMessageStore()
        messages_to_write = [
            ChatMessage.from_system("You are a helpful assistant. Answer the user's question."),
            ChatMessage.from_user("What is the capital of France?"),
            ChatMessage.from_assistant("The capital of France is Paris."),
        ]
        store.write_messages(index=index, messages=messages_to_write)

        pipe = Pipeline()
        pipe.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("{{ query }}")], required_variables=["query"]),
        )
        pipe.add_component("memory_retriever", ChatMessageRetriever(store))
        pipe.connect("prompt_builder.prompt", "memory_retriever.new_messages")

        res = pipe.run(
            data={"prompt_builder": {"query": "What is the capital of Germany?"}, "memory_retriever": {"index": index}}
        )
        assert res["memory_retriever"]["messages"] == [*messages_to_write, ChatMessage.from_user("What is the capital of Germany?")]

        # Clean up
        store.delete_messages(index=index)

    def test_chat_message_retriever_pipeline_serde(self):
        """
        Test that the ChatMessageRetriever can be used in a pipeline and that it can be serialized and deserialized.
        """
        pipe = Pipeline()
        pipe.add_component("memory_retriever", ChatMessageRetriever(InMemoryChatMessageStore()))

        # now serialize and deserialize the pipeline
        data = pipe.to_dict()
        new_pipe = Pipeline.from_dict(data)

        assert new_pipe == pipe

    def test_chat_message_store_with_chat_generator(self):
        store = InMemoryChatMessageStore()

        pipe = Pipeline()
        pipe.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("{{ query }}")], required_variables=["query"]),
        )
        pipe.add_component("message_retriever", ChatMessageRetriever(store))
        pipe.add_component("llm", MockChatGenerator())
        pipe.add_component("message_writer", ChatMessageWriter(store))

        pipe.connect("prompt_builder.prompt", "message_retriever.new_messages")
        pipe.connect("message_retriever.messages", "llm.messages")
        pipe.connect("llm.replies", "message_writer.messages")

        index = "user_123_session_456"
        result = pipe.run(
            data={
                "prompt_builder": {"query": "What is the capital of Germany?"},
                "message_retriever": {"index": index},
                "message_writer": {"index": index}
            },
            include_outputs_from={"llm"}
        )
        assert result["llm"]["replies"] == [ChatMessage.from_assistant("This is a mock response.")]
        # TODO Should improve example so the user message from prompt builder is also stored.
        assert store.retrieve_messages(index) == [ChatMessage.from_assistant("This is a mock response.")]

        # Clean up
        store.delete_messages(index)

    def test_chat_message_store_with_agent(self):
        store = InMemoryChatMessageStore()

        pipe = Pipeline()
        pipe.add_component(
            "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("{{ query }}")], required_variables=["query"]),
        )
        pipe.add_component("message_retriever", ChatMessageRetriever(store))
        pipe.add_component("agent", MockAgent())
        pipe.add_component("message_writer", ChatMessageWriter(store))

        pipe.connect("prompt_builder.prompt", "message_retriever.new_messages")
        pipe.connect("message_retriever.messages", "agent.messages")
        pipe.connect("agent.messages", "message_writer.messages")

        index = "user_123_session_456"
        result = pipe.run(
            data={
                "prompt_builder": {"query": "What is the capital of Germany?"},
                "message_retriever": {"index": index},
                "message_writer": {"index": index}
            },
            include_outputs_from={"agent"}
        )
        assert result["agent"]["last_message"] == ChatMessage.from_assistant("This is a mock response.")
        assert store.count_messages(index) == 2

        # Clean up
        store.delete_messages(index)
