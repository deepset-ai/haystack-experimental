
import pytest
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore


class TestChatMessageRetriever:
    def test_init(self):
        """
        Test that the ChatMessageRetriever component can be initialized with a valid message store.
        """
        message_store = InMemoryChatMessageStore()
        retriever = ChatMessageRetriever(message_store)

        assert retriever.message_store == message_store
        assert retriever.run() == {"messages": []}

    def test_retrieve_messages(self):
        """
        Test that the ChatMessageRetriever component can retrieve messages from the message store.
        """
        messages = [
            ChatMessage.from_user("Hello, how can I help you?"),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")
        ]

        message_store = InMemoryChatMessageStore()
        message_store.write_messages(messages)
        retriever = ChatMessageRetriever(message_store)

        assert retriever.message_store == message_store
        assert retriever.run() == {"messages": messages}

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
        message_store.write_messages(messages)
        retriever = ChatMessageRetriever(message_store)

        assert retriever.message_store == message_store
        assert retriever.run(last_k=1) == {
            "messages": [ChatMessage.from_user("Bonjour, comment puis-je vous aider?")]}

        assert retriever.run(last_k=2) == {
            "messages": [ChatMessage.from_user("Hola, como puedo ayudarte?"),
                         ChatMessage.from_user("Bonjour, comment puis-je vous aider?")
                         ]}

        # outliers
        assert retriever.run(last_k=10) == {
            "messages": [ChatMessage.from_user("Hello, how can I help you?"),
                         ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?"),
                         ChatMessage.from_user("Hola, como puedo ayudarte?"),
                         ChatMessage.from_user("Bonjour, comment puis-je vous aider?")
                         ]}

        with pytest.raises(ValueError):
            retriever.run(last_k=0)

        with pytest.raises(ValueError):
            retriever.run(last_k=-1)

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
        message_store.write_messages(messages)
        retriever = ChatMessageRetriever(message_store, last_k=2)

        assert retriever.message_store == message_store

        # last_k is 1 here from run parameter, overrides init of 2
        assert retriever.run(last_k=1) == {
            "messages": [ChatMessage.from_user("Bonjour, comment puis-je vous aider?")]}

        # last_k is 2 here from init
        assert retriever.run() == {
            "messages": [ChatMessage.from_user("Hola, como puedo ayudarte?"),
                         ChatMessage.from_user("Bonjour, comment puis-je vous aider?")
                         ]}

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
        """
        Test that the ChatMessageRetriever can be used in a pipeline and that it works as expected.
        """
        store = InMemoryChatMessageStore()
        store.write_messages([ChatMessage.from_assistant("Hello, how can I help you?")])

        pipe = Pipeline()
        pipe.add_component("memory_retriever", ChatMessageRetriever(store))
        pipe.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "memories"]))
        pipe.connect("memory_retriever", "prompt_builder.memories")
        user_prompt = """
        Given the following information, answer the question.

        Context:
        {% for memory in memories %}
            {{ memory.text }}
        {% endfor %}

        Question: {{ query }}
        Answer:
        """
        question = "What is the capital of France?"

        res = pipe.run(data={"prompt_builder": {"template": [ChatMessage.from_user(user_prompt)], "query": question}})
        resulting_prompt = res["prompt_builder"]["prompt"][0].text
        assert "France" in resulting_prompt
        assert "how can I help you" in resulting_prompt

    def test_chat_message_retriever_pipeline_serde(self):
        """
        Test that the ChatMessageRetriever can be used in a pipeline and that it can be serialized and deserialized.
        """
        pipe = Pipeline()
        pipe.add_component("memory_retriever", ChatMessageRetriever(InMemoryChatMessageStore()))
        pipe.add_component("prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user("no template")],
                                                               variables=["query"]))

        # now serialize and deserialize the pipeline
        data = pipe.to_dict()
        new_pipe = Pipeline.from_dict(data)

        assert new_pipe == pipe
