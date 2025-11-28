from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_experimental.memory.src.mem0.memory_store import Mem0MemoryStore

from haystack_experimental.components.memory_agents.agent import Agent

memory_store = Mem0MemoryStore(user_id="haystack_mem1")

messages = [
    ChatMessage.from_user("I like to listen to Russian pop music"),
    ChatMessage.from_user("I liked cold spanish latte with oat milk"),
    ChatMessage.from_user("I live in Florence Italy and I love mountains"),
    ChatMessage.from_user("""I am a software engineer and I like building application in python.
                                  Most of my projects are related to NLP and LLM agents.
                                  I find it easier to use Haystack framework to build my projects."""),
    ChatMessage.from_user("""I work in a startup and I am the CEO of the company.
                                  I have a team of 10 people and we are building a
                                  platform for small businesses to manage their customers and sales."""),
]

memory_store.add_memories(messages)
# memory_store.delete_all_memories()
result = memory_store.search_memories()

print(result)
