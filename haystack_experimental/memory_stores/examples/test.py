from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage

from haystack_experimental.components.memory_agents.agent import Agent
from haystack_experimental.memory.src.mem0.memory_store import Mem0MemoryStore

memory_store = Mem0MemoryStore(user_id="haystack_mem0")
memory_store.delete_all_memories()
