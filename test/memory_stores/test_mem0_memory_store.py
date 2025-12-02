# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import Mock, patch
import pytest
from time import sleep
from haystack.dataclasses.chat_message import ChatMessage

from haystack_experimental.memory_stores.mem0 import Mem0MemoryStore
from haystack.utils import Secret


class TestMem0MemoryStore:

    @pytest.fixture
    def mock_memory_client(self):
        """Mock the Mem0 MemoryClient."""
        with patch("haystack_experimental.memory_stores.mem0.src.mem0.memory_store.MemoryClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def mock_memory(self):
        """Mock the Mem0 Memory class."""
        with patch("haystack_experimental.memory_stores.mem0.src.mem0.memory_store.Memory") as mock_memory_class:
            mock_memory = Mock()
            mock_memory_class.from_config.return_value = mock_memory
            yield mock_memory

    @pytest.fixture
    def sample_messages(self):
        """Sample ChatMessage objects for testing."""
        return [
            ChatMessage.from_user("I usually work with Python language on LLM agents", meta={"source": "test"}),
            ChatMessage.from_user("I like working with Haystack. Show me how to make a simple RAG pipeline.", meta={"topic": "programming"}),
        ]

    @pytest.fixture
    def sample_conversation_with_assistant(self):
        return [
            ChatMessage.from_user("Suggest me some Italian food to cook at home."),
            ChatMessage.from_assistant("Nice! Do you prefer pasta dishes or something else?"),
            ChatMessage.from_user("Mostly pasta, especially anything with pesto."),
            ChatMessage.from_assistant("Here is a recipe of spicy Mediterranean pasta with pesto."),
            ChatMessage.from_user("Suggest me something not spicy."),
            ChatMessage.from_assistant("Here is a recipe of creamy mushroom pasta."),
            ChatMessage.from_user("What's the cooking time for this recipe?"),
        ]

    def test_init_with_user_id_and_api_key(self, mock_memory_client):
        """Test initialization with user_id and api_key."""
        with patch.dict(os.environ, {}, clear=True):
            store = Mem0MemoryStore(user_id="user123", api_key=Secret.from_token("test_api_key_12345"))
            assert store.user_id == "user123"
            assert store.run_id is None
            assert store.agent_id is None
            assert store.client == mock_memory_client
            assert store.search_criteria == None

    def test_init_with_params(self, mock_memory_client):
        search_criteria = {"query": "test query", "filters": {"category": "test"}, "top_k": 5}
        store = Mem0MemoryStore(
            user_id="user123", run_id="run456", agent_id="agent789", api_key=Secret.from_token("test_api_key_12345"), search_criteria=search_criteria
        )
        assert store.user_id == "user123"
        assert store.run_id == "run456"
        assert store.agent_id == "agent789"
        assert store.search_criteria == search_criteria
        assert store.client == mock_memory_client

    def test_init_with_memory_config(self, monkeypatch, mock_memory):
        """Test initialization with custom memory_config."""
        memory_config = {"llm": {"provider": "openai"}}
        with patch.dict(os.environ, {}, clear=True):
            monkeypatch.setenv("ENV_VAR", "test_api_key_12345")
            store = Mem0MemoryStore(user_id="user123", api_key=Secret.from_env_var("ENV_VAR"), memory_config=memory_config)
            assert store.memory_config == memory_config
            assert store.client == mock_memory

    def test_init_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="None of the following authentication environment variables are set"):
                Mem0MemoryStore(user_id="user123")

    def test_init_without_any_id_raises_error(self):
        """Test that initialization without any ID raises ValueError."""
        with pytest.raises(ValueError, match="At least one of user_id, run_id, or agent_id must be set"):
            Mem0MemoryStore(user_id=None, run_id=None, agent_id=None, api_key=Secret.from_token("test_api_key_12345"))

    def test_to_dict(self,monkeypatch, mock_memory_client):
        """Test serialization to dictionary."""
        with patch.dict(os.environ, {}, clear=True):
            monkeypatch.setenv("ENV_VAR", "test_api_key_12345")
            store = Mem0MemoryStore(user_id="user123", run_id="run456", agent_id="agent789", search_criteria={"top_k": 5}, api_key=Secret.from_env_var("ENV_VAR"))
            result = store.to_dict()
            assert result["init_parameters"]["user_id"] == "user123"
            assert result["init_parameters"]["run_id"] == "run456"
            assert result["init_parameters"]["agent_id"] == "agent789"
            assert result["init_parameters"]["api_key"] == {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"}
            assert result["init_parameters"]["search_criteria"] == {"top_k": 5}

    def test_from_dict(self, monkeypatch, mock_memory_client,):
        with patch.dict(os.environ, {}, clear=True):
            monkeypatch.setenv("ENV_VAR", "test_api_key_12345")
            data = {
                'type': 'haystack_experimental.memory_stores.mem0.src.mem0.memory_store.Mem0MemoryStore',
                'init_parameters': {'user_id': 'user123', 'run_id': 'run456',
                                    'agent_id': 'agent789', "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
                                    'memory_config': None, 'search_criteria': {'top_k': 5}}}
            store = Mem0MemoryStore.from_dict(data)
            assert store.user_id == "user123"
            assert store.run_id == "run456"
            assert store.agent_id == "agent789"
            assert store.client == mock_memory_client
            assert store.api_key == Secret.from_env_var("ENV_VAR")
            assert store.search_criteria == {"top_k": 5}

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_add_and_delete_memories(self, sample_messages):
        """Test adding memories successfully."""
        store = Mem0MemoryStore(user_id="haystack_test_123")
        # delete all memories for this id
        store.delete_all_memories(user_id="haystack_test_123")
        result = store.add_memories(sample_messages)
        assert len(result) == 2
        store.delete_all_memories(user_id="haystack_test_123")
        mem = store.search_memories()
        assert len(mem) == 0


    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_add_memories_with_infer_false(self, sample_messages):
        """Test adding memories with infer=False."""
        store = Mem0MemoryStore(user_id="haystack_test_123")
        # delete all memories for this id
        store.delete_all_memories(user_id="haystack_test_123")
        result = store.add_memories(sample_messages, infer=False)
        assert len(result) == 2

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_add_memories_with_metadata(self):
        """Test adding memories with metadata."""
        messages = [ChatMessage.from_user("Test", meta={"key": "value"})]
        store = Mem0MemoryStore(user_id="haystack_test_123")

        added_ids = store.add_memories(messages)
        assert added_ids[0] is not None

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_search_memories(self, sample_messages):
        """Test searching memories on previously added memories because the mem0 takes time to index the memory"""
        store = Mem0MemoryStore(user_id="haystack_simple_memories")

        # search without query
        result = store.search_memories()
        assert len(result) == 2

        # search with query
        result = store.search_memories(filters={"user_id": "haystack_query_memories"}, query="What programming languages do I usually work with?")
        assert result[0].text == "User likes working with python on NLP projects"

        # search with filters
        result = store.search_memories(filters={ "AND": [{"user_id": "haystack_query_memories"}, {"categories": {"in":["technology"]}}]})
        assert result[0].text == "User likes working with python on NLP projects"

        # search with metadata
        mem = store.search_memories(filters={ "AND": [{"user_id": "haystack_memories_with_metadata"}, {"metadata":   {"country": "Italy"}}]})
        assert mem[0].text == "User has visited Italy in 2025"
        assert mem[0].meta == {"country": "Italy", "timestamp": "04/2025"}

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_search_criteria_in_init(self):
        search_criteria = { "filters": {"AND": [{"user_id": "haystack_memories_with_metadata"}, {"metadata": {"timestamp": "04/2025"}}]}, "top_k": 1}
        store = Mem0MemoryStore(user_id="haystack_memories_with_metadata", search_criteria=search_criteria)
        result = store.search_memories()
        assert len(result) == 1
        assert result[0].text == "User has visited Italy in 2025"
        assert result[0].meta == {"country": "Italy", "timestamp": "04/2025"}

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_delete_all_memories(self):
        """Test deleting all memories."""
        store = Mem0MemoryStore(user_id="haystack_test_123")
        store.delete_all_memories()

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_delete_memory(self, sample_messages):
        """Test deleting a single memory."""
        store = Mem0MemoryStore(user_id="haystack_test_123")
        store.delete_all_memories(user_id="haystack_test_123")
        store.add_memories(sample_messages, infer=False)
        sleep(10)
        mem = store.search_memories(_include_memory_metadata=True)
        store.delete_memory(mem[0].meta["id"])
        sleep(10)
        assert len(store.search_memories()) == 1

    @pytest.mark.skipif(
        not os.environ.get("MEM0_API_KEY", None),
        reason="Export an env var called MEM0_API_KEY containing the Mem0 API key to run this test.",
    )
    @pytest.mark.integration
    def test_role_based_memories(self):
        messages = [
            ChatMessage.from_user("I'm planning to watch a movie tonight. Any recommendations?"),
            ChatMessage.from_assistant("How about thriller movies? They can be quite engaging."),
            ChatMessage.from_user("I'm not a big fan of thriller movies but I love sci-fi movies."),
            ChatMessage.from_assistant("Got it! Then I would recommend Interstellar or Inception? I would also recommend watching some Japanese anime movies."),
        ]
        store = Mem0MemoryStore(user_id="haystack_role_based_memories", agent_id="movie_agent")
        store.delete_all_memories(user_id="haystack_role_based_memories")
        store.delete_all_memories(agent_id="movie_agent")
        sleep(10)
        store.add_memories(messages, infer=False)
        assistant_mem = store.search_memories(filters={"agent_id": "movie_agent"})
        user_mem = store.search_memories(filters={"user_id": "haystack_role_based_memories"})
        assert len(assistant_mem) == 2
        assert len(user_mem) == 2


    def test_get_scope_with_only_user_id(self, mock_memory_client):
        """Test _get_scope returns only user_id when others are None."""
        with patch.dict(os.environ, {}, clear=True):
            store = Mem0MemoryStore(user_id="user123", api_key=Secret.from_token("test_api_key_12345"))
            scope = store._get_ids()
            assert scope == {"user_id": "user123"}
