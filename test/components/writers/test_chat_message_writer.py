from haystack.dataclasses import ChatMessage

from haystack_experimental.components.writers import ChatMessageWriter
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore


class TestChatMessageRetriever:
    def test_init(self):
        messages = [
            ChatMessage.from_user(content="Hello, how can I help you?"),
            ChatMessage.from_user(content="Hallo, wie kann ich Ihnen helfen?")
        ]

        message_store = InMemoryChatMessageStore()
        message_store.write_messages(messages)
        retriever = ChatMessageWriter(message_store)

        assert retriever.message_store == message_store
        assert retriever.run(messages=messages) == {"messages_written": 2}

    def test_to_dict(self):
        message_store = InMemoryChatMessageStore()
        retriever = ChatMessageWriter(message_store)

        data = retriever.to_dict()
        assert data == {
            "type": "haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter",
            "init_parameters": {
                "message_store": {
                    "init_parameters": {},
                    "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
                }
            },
        }

    def test_from_dict(self):
        data = {
            "type": "haystack_experimental.components.writers.chat_message_writer.ChatMessageWriter",
            "init_parameters": {
                "message_store": {
                    "init_parameters": {},
                    "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
                }
            },
        }
        retriever = ChatMessageWriter.from_dict(data)
        assert retriever.message_store.to_dict() == {
            "init_parameters": {},
            "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
        }
