
"""
Simple Agent Memory Example - Direct Pipeline Integration

This example shows a straightforward integration of memory components with an Agent
in a Haystack pipeline, similar to the database assistant example structure.

Pipeline: MemoryRetriever -> ChatPromptBuilder -> Agent -> MemoryWriter
"""

import os
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.core.pipeline import Pipeline
from haystack.tools import tool
from haystack.dataclasses import ChatMessage
from haystack_experimental.memory import Mem0MemoryRetriever, MemoryWriter, Mem0MemoryStore


@tool
def save_user_preference(preference_type: str, preference_value: str) -> str:
    """Save user preferences that should be remembered"""
    return f"✅ Saved preference: {preference_type} = {preference_value}"


@tool
def get_recommendation(category: str) -> str:
    """Get personalized recommendations based on user preferences"""
    recommendations = {
        "food": "Based on your preferences, try the Mediterranean cuisine!",
        "music": "I recommend some jazz playlists for you!",
        "books": "You might enjoy science fiction novels!",
    }
    return recommendations.get(category, "I'll learn your preferences to give better recommendations!")


# Create memory store
memory_store = Mem0MemoryStore(api_key=os.getenv("MEM0_API_KEY"))

# Create memory-aware agent
memory_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[save_user_preference, get_recommendation],
    system_prompt="""
    You are a personal assistant with memory capabilities.
    Use the provided memories to personalize your responses and remember user context.
    When users share preferences, use the save_user_preference tool.
    When asked for recommendations, use the get_recommendation tool.
    Be conversational and reference previous interactions when relevant.
    """,
    exit_conditions=["text"],
    max_agent_steps=10,
    raise_on_tool_invocation_failure=False
)

# Create the pipeline
agent_memory_pipeline = Pipeline()

# Add components
agent_memory_pipeline.add_component("memory_retriever", Mem0MemoryRetriever(
    memory_store=memory_store,
    top_k=5
))

agent_memory_pipeline.add_component("prompt_builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_system(
            "Previous conversation context:\n"
            "{% for memory in memories %}"
            "{{ memory.content }}\n"
            "{% endfor %}"
            "{% if not memories %}No previous context available.{% endif %}"
        ),
        ChatMessage.from_user("{{ user_query }}")
    ],
    required_variables=["user_query"]
))

agent_memory_pipeline.add_component("agent", memory_agent)
agent_memory_pipeline.add_component("memory_writer", MemoryWriter(memory_store=memory_store))

# Connect components
agent_memory_pipeline.connect("memory_retriever.memories", "prompt_builder.memories")
agent_memory_pipeline.connect("prompt_builder.prompt", "agent.messages")
agent_memory_pipeline.connect("agent.messages", "memory_writer.messages")

# Run the pipeline
user_id = "alice_123"
user_query = "Can you remember this and give me a food recommendation?"

# Get memories and run agent
agent_output = agent_memory_pipeline.run({
    "memory_retriever": {
        "query": user_query,
        "user_id": user_id
    },
    "prompt_builder": {
        "user_query": user_query
    },
    "memory_writer": {
        "user_id": user_id
    }
})
