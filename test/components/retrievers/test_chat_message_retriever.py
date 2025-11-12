import pytest
from typing import Any, Optional

from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.writers import ChatMessageWriter


@pytest.fixture
def store():
    msg_store = InMemoryChatMessageStore()
    yield msg_store
    msg_store.delete_all_messages()


@component
class MockChatGenerator:
    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]) -> dict[str, list[ChatMessage]]:
        return {"replies": [ChatMessage.from_assistant("This is a mock response.")]}


@component
class MockAgent:
    def __init__(self, system_prompt: Optional[str] = None):
        self.system_prompt = system_prompt

    @component.output_types(messages=list[ChatMessage], last_message=ChatMessage)
    def run(self, messages: list[ChatMessage]) -> dict[str, Any]:
        if self.system_prompt:
            system_msg = ChatMessage.from_system(self.system_prompt)
            messages = [system_msg, *messages]

        assistant_msg = ChatMessage.from_assistant("This is a mock response.")
        return {"messages": [*messages, assistant_msg], "last_message": assistant_msg}


class TestChatMessageRetriever:
    def test_init(self, store):
        """
        Test that the ChatMessageRetriever component can be initialized with a valid message store.
        """
        retriever = ChatMessageRetriever(store)
        assert retriever.message_store == store
        assert retriever.run(index="test") == {"messages": []}

    def test_retrieve_messages(self, store):
        """
        Test that the ChatMessageRetriever component can retrieve messages from the message store.
        """
        messages = [
            ChatMessage.from_user("Hello, how can I help you?"),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?")
        ]
        store.write_messages(index="test", messages=messages)
        retriever = ChatMessageRetriever(store)
        assert retriever.message_store == store
        assert retriever.run(index="test") == {"messages": messages}

    def test_retrieve_messages_last_k(self, store):
        """
        Test that the ChatMessageRetriever component can retrieve last_k messages from the message store.
        """
        messages = [
            ChatMessage.from_user("Hello, how can I help you?"),
            ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?"),
            ChatMessage.from_user("Hola, como puedo ayudarte?"),
            ChatMessage.from_user("Bonjour, comment puis-je vous aider?")
        ]

        store.write_messages(index="test", messages=messages)
        retriever = ChatMessageRetriever(store)

        assert retriever.message_store == store
        assert retriever.run(index="test", last_k=1) == {
            "messages": [ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"chat_message_id": "3"})]
        }

        assert retriever.run(index="test", last_k=2) == {
            "messages": [
                ChatMessage.from_user("Hola, como puedo ayudarte?", meta={"chat_message_id": "2"}),
                ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"chat_message_id": "3"}),
            ]
        }

        # outliers
        assert retriever.run(index="test", last_k=10) == {
            "messages": [
                ChatMessage.from_user("Hello, how can I help you?", meta={"chat_message_id": "0"}),
                ChatMessage.from_user("Hallo, wie kann ich Ihnen helfen?", meta={"chat_message_id": "1"}),
                ChatMessage.from_user("Hola, como puedo ayudarte?", meta={"chat_message_id": "2"}),
                ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"chat_message_id": "3"}),
            ]
        }

        with pytest.raises(ValueError):
            retriever.run(index="test", last_k=0)

        with pytest.raises(ValueError):
            retriever.run(index="test", last_k=-1)

    def test_retrieve_messages_last_k_init(self, store):
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

        store.write_messages(index="test", messages=messages)
        retriever = ChatMessageRetriever(store, last_k=2)

        assert retriever.message_store == store

        # last_k is 1 here from run parameter, overrides init of 2
        assert retriever.run(index="test", last_k=1) == {
            "messages": [ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"chat_message_id": "3"})]
        }

        # last_k is 2 here from init
        assert retriever.run(index="test") == {
            "messages": [
                ChatMessage.from_user("Hola, como puedo ayudarte?", meta={"chat_message_id": "2"}),
                ChatMessage.from_user("Bonjour, comment puis-je vous aider?", meta={"chat_message_id": "3"}),
            ]
        }

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

    def test_chat_message_retriever_pipeline(self, store):
        index = "user_123_session_456"
        messages_to_write = [
            ChatMessage.from_user("What is the capital of France?"),
            ChatMessage.from_assistant("The capital of France is Paris."),
        ]
        store.write_messages(index=index, messages=messages_to_write)

        pipe = Pipeline()
        pipe.add_component(
            "prompt_builder", ChatPromptBuilder(
                template=[
                    ChatMessage.from_system("You are a helpful assistant. Answer the user's question."),
                    ChatMessage.from_user("{{ query }}")
                ],
                required_variables=["query"]
            ),
        )
        pipe.add_component("memory_retriever", ChatMessageRetriever(store))
        pipe.connect("prompt_builder.prompt", "memory_retriever.new_messages")

        res = pipe.run(
            data={"prompt_builder": {"query": "What is the capital of Germany?"}, "memory_retriever": {"index": index}}
        )
        assert res["memory_retriever"]["messages"] == [
            ChatMessage.from_system("You are a helpful assistant. Answer the user's question."),
            ChatMessage.from_user("What is the capital of France?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("The capital of France is Paris.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Germany?")
        ]

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

    def test_chat_message_store_with_chat_generator(self, store):
        pipe = Pipeline()
        pipe.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=[ChatMessage.from_system("This is a system prompt."), ChatMessage.from_user("{{ query }}")],
                required_variables=["query"]
            ),
        )
        pipe.add_component("message_retriever", ChatMessageRetriever(store))
        pipe.add_component("llm", MockChatGenerator())
        pipe.add_component(
            "message_joiner",
            OutputAdapter(template="{{ prompt + replies }}", output_type=list[ChatMessage], unsafe=True)
        )
        pipe.add_component("message_writer", ChatMessageWriter(store))

        pipe.connect("prompt_builder.prompt", "message_retriever.new_messages")
        pipe.connect("message_retriever.messages", "llm.messages")
        pipe.connect("prompt_builder.prompt", "message_joiner.prompt")
        pipe.connect("llm.replies", "message_joiner.replies")
        pipe.connect("message_joiner", "message_writer.messages")

        index = "user_123_session_456"
        result = pipe.run(
            data={
                "prompt_builder": {"query": "What is the capital of Germany?"},
                "message_retriever": {"index": index},
                "message_writer": {"index": index}
            },
            include_outputs_from={"message_retriever", "llm"}
        )
        assert result["message_retriever"]["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?"),
        ]
        assert result["llm"]["replies"] == [ChatMessage.from_assistant("This is a mock response.")]
        # We don't expect the system prompt to be stored b/c InMemoryChatMessageStore defaults to
        # skip_system_messages=True
        assert store.retrieve_messages(index) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"})
        ]

        # Second run to verify that retrieval works as expected
        result = pipe.run(
            data={
                "prompt_builder": {"query": "What is the capital of Italy?"},
                "message_retriever": {"index": index},
                "message_writer": {"index": index}
            },
            include_outputs_from={"message_retriever", "llm"}
        )
        # Check that the retrieved messages include all previous messages and that the new user message is appended
        # and the system prompt is still at the beginning.
        assert result["message_retriever"]["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            # The new user message doesn't have a chat_message_id yet. It's assigned when written to the store.
            ChatMessage.from_user("What is the capital of Italy?"),
        ]
        assert result["llm"]["replies"] == [ChatMessage.from_assistant("This is a mock response.")]
        assert store.retrieve_messages(index) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Italy?", meta={"chat_message_id": "2"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "3"}),
        ]

    def test_chat_message_store_with_agent(self, store):
        store = InMemoryChatMessageStore()

        pipe = Pipeline()
        pipe.add_component(
            "prompt_builder",
            ChatPromptBuilder(template=[ChatMessage.from_user("{{ query }}")], required_variables=["query"]),
        )
        pipe.add_component("message_retriever", ChatMessageRetriever(store))
        pipe.add_component("agent", MockAgent(system_prompt="This is a system prompt."))
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
        assert result["agent"]["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?"),
            ChatMessage.from_assistant("This is a mock response."),
        ]
        assert store.retrieve_messages(index) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
        ]

        # Second run
        result = pipe.run(
            data={
                "prompt_builder": {"query": "What is the capital of Italy?"},
                "message_retriever": {"index": index},
                "message_writer": {"index": index}
            },
            include_outputs_from={"agent"}
        )
        assert result["agent"]["messages"] == [
            ChatMessage.from_system("This is a system prompt."),
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Italy?"),
            ChatMessage.from_assistant("This is a mock response."),
        ]
        assert store.retrieve_messages(index) == [
            ChatMessage.from_user("What is the capital of Germany?", meta={"chat_message_id": "0"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "1"}),
            ChatMessage.from_user("What is the capital of Italy?", meta={"chat_message_id": "2"}),
            ChatMessage.from_assistant("This is a mock response.", meta={"chat_message_id": "3"}),
        ]


class TestChatMessageRetrieveLastK:
    @pytest.mark.parametrize("last_k, start_idx", [(1, 4), (2, 2), (3, 1), (4, 0), (5, 0)])
    def test_last_k_agent_history(self, last_k, start_idx):
        tool_call = ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        # System, User, ToolCall, ToolOutput, Assistant
        messages = [
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_tool(tool_result="It's sunny in Paris.", origin=tool_call),
            ChatMessage.from_assistant("The weather in Paris is sunny."),
        ]
        retriever = ChatMessageRetriever(InMemoryChatMessageStore())

        last_k_messages = retriever._get_last_k_messages(messages, last_k)

        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k, start_idx", [(1, 5), (2, 2), (3, 1), (4, 0), (5, 0), (6, 0)])
    def test_last_k_agent_history_with_two_tool_calls(self, last_k, start_idx):
        tool_call_1 = ToolCall(tool_name="get_weather", arguments={"location": "Paris"})
        tool_call_2 = ToolCall(tool_name="get_time", arguments={"location": "Paris"})
        # System, User, 2 ToolCalls, ToolOutput, ToolOutput, Assistant
        messages = [
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call_1, tool_call_2]),
            ChatMessage.from_tool(tool_result="It's sunny in Paris.", origin=tool_call_1),
            ChatMessage.from_tool(tool_result="It's 3 PM in Paris.", origin=tool_call_2),
            ChatMessage.from_assistant("The weather in Paris is sunny and it's 3 PM there."),
        ]
        retriever = ChatMessageRetriever(InMemoryChatMessageStore())

        last_k_messages = retriever._get_last_k_messages(messages, last_k)
        assert last_k_messages == messages[start_idx:]

    @pytest.mark.parametrize("last_k,start_idx", [(1, 5), (2, 2), (3, 1), (4, 0), (5, 0), (6, 0)])
    def test_last_k_agent_history_with_intervening_user_message(self, last_k, start_idx):
        """Relevant for simulating a human-in-the-loop scenario where the user modifies parameters"""
        tool_call = ToolCall(tool_name="get_weather", arguments={"location": "Paris"}, id="tc_1")
        modified_tool_call = ToolCall(tool_name="get_weather", arguments={"location": "London"}, id="tc_1")
        # System, User, ToolCall, User, ToolOutput, Assistant
        messages = [
            ChatMessage.from_system("You are a helpful assistant."),
            ChatMessage.from_user("What is the weather in Paris?"),
            ChatMessage.from_assistant(tool_calls=[tool_call]),
            ChatMessage.from_user("The user modified the tool parameters to {'location': 'London'}."),
            ChatMessage.from_tool(tool_result="It's sunny in London.", origin=modified_tool_call),
            ChatMessage.from_assistant("The weather in London is sunny."),
        ]
        retriever = ChatMessageRetriever(InMemoryChatMessageStore())

        last_k_messages = retriever._get_last_k_messages(messages, last_k=last_k)
        assert last_k_messages == messages[start_idx:]
