# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch
import pytest
import uuid
from time import sleep
from haystack.dataclasses.chat_message import ChatMessage
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack_experimental.components.agents.agent import Agent
from haystack_experimental.memory_stores.mem0 import Mem0MemoryStore
from haystack.utils import Secret

def _get_unique_user_id() -> str:
    """
    Generate a unique, valid user_id for test isolation in memory store.

    Each test gets its own index to enable parallel test execution without conflicts.
    """
    return f"test_{uuid.uuid4().hex}"

@pytest.fixture
def memory_store():
    """
    We use this memory store for basic tests and for testing filters.
    """
    store = Mem0MemoryStore()
    user_id = _get_unique_user_id()
    yield store, user_id

    assert store.client
    store.delete_all_memories(user_id=user_id)

class TestMem0MemoryStore:

    @pytest.fixture
    def sample_messages(self):
        """Sample ChatMessage objects for testing."""
        return [
            ChatMessage.from_user("I usually work with Python language on LLM agents", meta={"source": "test"}),
            ChatMessage.from_user("I like working with Haystack.", meta={"topic": "programming"}),
        ]

    def test_init_with_user_id_and_api_key(self, mock_memory_client):
        """Test initialization with user_id and api_key."""
        with patch.dict(os.environ, {}, clear=True):
            store = Mem0MemoryStore(api_key=Secret.from_token("test_api_key_12345"))
            assert store.client == mock_memory_client

    def test_init_with_params(self, mock_memory_client):
        store = Mem0MemoryStore(
            api_key=Secret.from_token("test_api_key_12345")
        )
        assert store.client == mock_memory_client

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError,
                               match="None of the following authentication environment variables are set"):
                Mem0MemoryStore()

    def test_to_dict(self, monkeypatch, mock_memory_client):
        """Test serialization to dictionary."""
        with patch.dict(os.environ, {}, clear=True):
            monkeypatch.setenv("ENV_VAR", "test_api_key_12345")
            store = Mem0MemoryStore(
                api_key=Secret.from_env_var("ENV_VAR"))

            result = store.to_dict()
            assert result["init_parameters"]["api_key"] == {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"}

    def test_from_dict(self, monkeypatch, mock_memory_client):
        with patch.dict(os.environ, {}, clear=True):
            monkeypatch.setenv("ENV_VAR", "test_api_key_12345")
            data = {
                'type': 'haystack_experimental.memory_stores.mem0.memory_store.Mem0MemoryStore',
                'init_parameters': { "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                                    }}
            store = Mem0MemoryStore.from_dict(data)
            assert store.client == mock_memory_client
            assert store.api_key == Secret.from_env_var("ENV_VAR")

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_add_memories(self, sample_messages, memory_store):
        """Test adding memories successfully."""
        store, user_id = memory_store
        result = store.add_memories(messages=sample_messages, user_id=user_id)
        assert len(result) == 2

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_add_memories_with_infer_false(self, sample_messages, memory_store):
        """Test adding memories with infer=False."""
        store, user_id = memory_store
        result = store.add_memories(messages=sample_messages, infer=False, user_id=user_id)
        assert len(result) == 2

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_add_memories_with_metadata(self, memory_store):
        """Test adding memories with metadata."""
        store, user_id = memory_store
        messages = [ChatMessage.from_user("User likes to work with python on NLP projects")]
        result = store.add_memories(
            messages=messages, user_id=user_id, metadata={"key": "value"}, async_mode=False
        )
        assert len(result) == 1

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_search_memories(self, sample_messages):
        """Test searching memories on previously added memories because the mem0 takes time to index the memory"""
        memory_store = Mem0MemoryStore()
        # search without query
        result = memory_store.search_memories(user_id="haystack_simple_memories")
        assert len(result) == 2

        # search with query
        result = memory_store.search_memories(
            user_id="haystack_query_memories",
            query="What programming languages do I usually work with?",
            include_memory_metadata=True
        )
        assert result[0].text == "User likes working with python on NLP projects"

        # search with filters
        result = memory_store.search_memories( filters={
            "operator": "AND",
            "conditions": [
                {"field": "user_id", "operator": "==", "value": "haystack_query_memories"},
                {"field": "categories", "operator": "in", "value": ["technology"]},
            ],
        })
        assert result[0].text == "User likes working with python on NLP projects"

        # search with metadata
        mem = memory_store.search_memories(filters={
            "operator": "AND",
            "conditions": [
                {"field": "user_id", "operator": "==", "value": "haystack_memories_with_metadata"},
                {"field": "metadata", "operator": "==", "value": {"country": "Italy"}},
                 ]},

        )
        assert mem[0].text == "User has visited Italy in 2025"
        assert mem[0].meta == {"country": "Italy", "timestamp": "04/2025"}

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_search_memories_as_single_message(self):
        """Test searching memories as a single message."""
        memory_store = Mem0MemoryStore()
        result = memory_store.search_memories_as_single_message(user_id="haystack_simple_memories")
        assert result.text is not None
        assert len(result) == 1

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_delete_memory(self, sample_messages, memory_store):
        """Test deleting a single memory."""
        store, user_id = memory_store
        store.delete_all_memories(user_id=user_id)
        store.add_memories(messages=sample_messages, infer=False, user_id=user_id)
        sleep(10)
        mem = store.search_memories(user_id=user_id, include_memory_metadata=True)
        store.delete_memory(memory_id=mem[0].meta["retrieved_memory_metadata"]["id"])
        sleep(10)
        assert len(store.search_memories(user_id=user_id)) == 1

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_role_based_memories(self, memory_store):
        store, user_id = memory_store
        unique_agent_id = _get_unique_user_id()
        messages = [
            ChatMessage.from_user("I'm planning to watch a movie tonight. Any recommendations?"),
            ChatMessage.from_assistant("How about thriller movies? They can be quite engaging."),
            ChatMessage.from_user("I'm not a big fan of thriller movies but I love sci-fi movies."),
            ChatMessage.from_assistant(
                "Got it! Then I would recommend Interstellar or Inception? I would also recommend watching some "
                "Japanese anime movies."
            ),
        ]
        store.add_memories(messages=messages, infer=False, user_id=user_id, agent_id=unique_agent_id)
        assistant_mem = store.search_memories(filters={"field": "agent_id", "operator": "==", "value": unique_agent_id})
        user_mem = store.search_memories(filters={"field": "user_id", "operator": "==", "value": user_id})
        assert len(assistant_mem) == 2
        assert len(user_mem) == 2

    @pytest.mark.skipif(
        not (os.environ.get("MEM0_API_KEY", None) and os.environ.get("OPENAI_API_KEY", None)),
        reason="Export an env var called MEM0_API_KEY and OPENAI_API_KEY containing the Mem0 API key and OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_memory_store_with_agent(self, memory_store):
        store, user_id = memory_store
        messages = [ChatMessage.from_user("User likes to work with python on NLP projects")]
        _ = store.add_memories(messages=messages, user_id=user_id, async_mode=False)
        sleep(10)
        agent = Agent(chat_generator=OpenAIChatGenerator(), memory_store=store)
        answer = agent.run(
            messages=[ChatMessage.from_user("Based on what you know about me, what programming language I work with?")],
            memory_store_kwargs={"user_id": user_id},
        )
        assert answer is not None
        assert "python" in answer["last_message"].text.lower()
