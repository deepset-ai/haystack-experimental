from haystack.dataclasses import ChatMessage

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore


class TestChatMessageStore:

    def test_init(self):
        store = InMemoryChatMessageStore()
        assert store.count_messages() == 0
        assert store.retrieve() == []
        assert store.write_messages([]) == 0
        assert not store.delete_messages()

    def test_to_dict(self):
        store = InMemoryChatMessageStore()
        assert store.to_dict() == {
            "init_parameters": {},
            "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
        }

    def test_from_dict(self):
        data = {
            "init_parameters": {},
            "type": "haystack_experimental.chat_message_stores.in_memory.chat_message_store.InMemoryChatMessageStore"
        }
        store = InMemoryChatMessageStore.from_dict(data)
        assert store.to_dict() == data

    def test_count_messages(self):
        store = InMemoryChatMessageStore()
        assert store.count_messages() == 0
        store.write_messages(messages=[ChatMessage.from_user(content="Hello, how can I help you?")])
        assert store.count_messages() == 1
        store.write_messages(messages=[ChatMessage.from_user(content="Hallo, wie kann ich Ihnen helfen?")])
        assert store.count_messages() == 2
        store.write_messages(messages=[ChatMessage.from_user(content="Hola, ¿cómo puedo ayudarte?")])
        assert store.count_messages() == 3

    def test_retrieve(self):
        store = InMemoryChatMessageStore()
        assert store.retrieve() == []
        store.write_messages(messages=[ChatMessage.from_user(content="Hello, how can I help you?")])
        assert store.retrieve() == [ChatMessage.from_user(content="Hello, how can I help you?")]
        store.write_messages(messages=[ChatMessage.from_user(content="Hallo, wie kann ich Ihnen helfen?")])
        assert store.retrieve() == [
            ChatMessage.from_user(content="Hello, how can I help you?"),
            ChatMessage.from_user(content="Hallo, wie kann ich Ihnen helfen?"),
        ]
        store.write_messages(messages=[ChatMessage.from_user(content="Hola, ¿cómo puedo ayudarte?")])
        assert store.retrieve() == [
            ChatMessage.from_user(content="Hello, how can I help you?"),
            ChatMessage.from_user(content="Hallo, wie kann ich Ihnen helfen?"),
            ChatMessage.from_user(content="Hola, ¿cómo puedo ayudarte?"),
        ]

    def test_delete_messages(self):
        store = InMemoryChatMessageStore()
        assert store.count_messages() == 0
        store.write_messages(messages=[ChatMessage.from_user(content="Hello, how can I help you?")])
        assert store.count_messages() == 1
        store.delete_messages()
        assert store.count_messages() == 0
        store.write_messages(messages=[ChatMessage.from_user(content="Hallo, wie kann ich Ihnen helfen?")])
        store.write_messages(messages=[ChatMessage.from_user(content="Hola, ¿cómo puedo ayudarte?")])
        assert store.count_messages() == 2
        store.delete_messages()
        assert store.count_messages() == 0
