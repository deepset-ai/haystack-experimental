import pytest

from haystack.dataclasses import ChatMessage

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore


@pytest.fixture
def store():
    msg_store = InMemoryChatMessageStore()
    yield msg_store
    msg_store.delete_all_messages()


class TestInMemoryChatMessageStore:

    def test_init(self, store):
        """
        Test that the InMemoryChatMessageStore can be initialized and that it works as expected.
        """
        assert store.count_messages(index="test") == 0
        assert store.retrieve_messages(index="test") == []
        assert store.write_messages(index="test", messages=[]) == 0
        assert not store.delete_messages(index="test")

    def test_to_dict(self):
        """
        Test that the InMemoryChatMessageStore can be serialized to a dictionary.
        """
        store = InMemoryChatMessageStore()
        assert store.to_dict() == {
            "init_parameters": {
                "skip_system_messages": True,
                "last_k": 10
            },
            "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"
        }

    def test_from_dict(self):
        """
        Test that the InMemoryChatMessageStore can be deserialized from a dictionary.
        """
        data = {
            "init_parameters": {
                "skip_system_messages": True,
                "last_k": 10
            },
            "type": "haystack_experimental.chat_message_stores.in_memory.InMemoryChatMessageStore"
        }
        store = InMemoryChatMessageStore.from_dict(data)
        assert store.to_dict() == data

    def test_count_messages(self, store):
        """
        Test that the InMemoryChatMessageStore can count the number of messages in the store correctly.
        """
        assert store.count_messages(index="test") == 0
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        assert store.count_messages(index="test") == 1
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")])
        assert store.count_messages(index="test") == 2
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?")])
        assert store.count_messages(index="test") == 3

    def test_retrieve(self, store):
        """
        Test that the InMemoryChatMessageStore can retrieve all messages from the store correctly.
        """
        assert store.retrieve_messages(index="test") == []
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        assert store.retrieve_messages(index="test") == [
            ChatMessage.from_user("Hello, how can I help you?", meta={"chat_message_id": "0"}),
        ]
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")])
        assert store.retrieve_messages(index="test") == [
            ChatMessage.from_user("Hello, how can I help you?", meta={"chat_message_id": "0"}),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?", meta={"chat_message_id": "1"}),
        ]
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?")])
        assert store.retrieve_messages(index="test") == [
            ChatMessage.from_user("Hello, how can I help you?", meta={"chat_message_id": "0"}),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?", meta={"chat_message_id": "2"}),
        ]

    def test_delete_messages(self, store):
        """
        Test that the InMemoryChatMessageStore can delete all messages from the store correctly.
        """
        assert store.count_messages(index="test") == 0
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hello, how can I help you?")])
        assert store.count_messages(index="test") == 1
        store.delete_messages(index="test")
        assert store.count_messages(index="test") == 0
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")])
        store.write_messages(index="test", messages=[ChatMessage.from_user("Hola, ¿cómo puedo ayudarte?")])
        assert store.count_messages(index="test") == 2
        store.delete_messages(index="test")
        assert store.count_messages(index="test") == 0
