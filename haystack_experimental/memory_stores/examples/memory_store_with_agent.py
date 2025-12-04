# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.generators.chat.openai import OpenAIChatGenerator

from haystack.dataclasses import ChatMessage

from haystack_experimental.components.memory_agents.agent import Agent
from haystack_experimental.memory_stores.mem0.memory_store import Mem0MemoryStore

memory_store = Mem0MemoryStore(user_id="haystack_mem0",
                               search_criteria={"top_k": 3})

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

chat_generator = OpenAIChatGenerator()
agent = Agent(chat_generator=chat_generator, memory_store=memory_store)
answer = agent.run(messages=[ChatMessage.from_user("Based on what you know about me, what programming language I work with?")])

print("Replies: ", answer)
