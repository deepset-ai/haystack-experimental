from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore


class TestChatMessageRetriever:
    def test_init(self):

        messages = [
            ChatMessage.from_user(content="Hello, how can I help you?"),
            ChatMessage.from_user(content="Hallo, wie kann ich Ihnen helfen?")
        ]

        message_store = InMemoryChatMessageStore()
        message_store.write_messages(messages)
        retriever = ChatMessageRetriever(message_store)

        assert retriever.message_store == message_store
        assert retriever.run() == {"messages": messages}

    def test_to_dict(self):
        message_store = InMemoryChatMessageStore()
        retriever = ChatMessageRetriever(message_store)

        data = retriever.to_dict()
        assert data == {
            "type": "haystack_experimental.components.retrievers.chat_message_retriever.ChatMessageRetriever",
            "init_parameters": {
                "message_store": {
                    "init_parameters": {},
                    "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
                }
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_experimental.components.retrievers.chat_message_retriever.ChatMessageRetriever",
            "init_parameters": {
                "message_store": {
                    "init_parameters": {},
                    "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
                }
            },
        }
        retriever = ChatMessageRetriever.from_dict(data)
        assert retriever.message_store.to_dict() == {
            "init_parameters": {},
            "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
        }

    def test_chat_message_retriever_pipeline(self):
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
            {{ memory.content }}
        {% endfor %}

        Question: {{ query }}
        Answer:
        """
        question = "What is the capital of France?"

        res = pipe.run(data={"prompt_builder": {"template": [ChatMessage.from_user(user_prompt)], "query": question}})
        resulting_prompt = res["prompt_builder"]["prompt"][0].content
        assert "France" in resulting_prompt
        assert "how can I help you" in resulting_prompt
