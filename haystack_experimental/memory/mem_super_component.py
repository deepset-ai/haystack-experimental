from haystack import Pipeline, SuperComponent, super_component
from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat import ChatGenerator
from haystack.dataclasses import ChatMessage, Document
from haystack.tools import Tool

from .mem0_store import Mem0MemoryStore
from .memory_retriever import Mem0MemoryRetriever
from .memory_writer import MemoryWriter


@super_component
class AgentMemory:
    def __init__(
        self,
        query: str,
        user_id: str,
        system_prompt: str,
        tools: list[Tool],
        chat_generator: ChatGenerator,
        exit_conditions: list[str],
        max_agent_steps: int,
        raise_on_tool_invocation_failure: bool,
    ):
        memory_store = Mem0MemoryStore(user_id=user_id)
        memory_retriever = Mem0MemoryRetriever(memory_store=memory_store)
        memory_writer = MemoryWriter(memory_store=memory_store)
        agent = Agent(
            chat_generator=chat_generator,
            tools=tools,
            system_prompt=system_prompt,
            exit_conditions=exit_conditions,
            max_agent_steps=max_agent_steps,
            raise_on_tool_invocation_failure=raise_on_tool_invocation_failure,
        )
        pipeline = Pipeline()
        pipeline.add_component("memory_retriever", memory_retriever)

        pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                template=[
                    ChatMessage.from_system(
                        "Previous conversation context:\n"
                        "{% for memory in memories %}"
                        "{{ memory.content }}\n"
                        "{% endfor %}"
                        "{% if not memories %}No previous context available.{% endif %}"
                    ),
                    ChatMessage.from_user("{{ user_query }}"),
                ],
                required_variables=["user_query"],
            ),
        )

        pipeline.add_component("agent", agent)
        pipeline.add_component("memory_writer", memory_writer)

        # Connect components
        pipeline.connect("memory_retriever.memories", "prompt_builder.memories")
        pipeline.connect("prompt_builder.prompt", "agent.messages")
        pipeline.connect("agent.messages", "memory_writer.messages")

        self.output_mapping = {
            "agent.messages": "messages",
            "memory_writer.messages": "messages",
        }
        self.input_mapping = {
            "query": "retriever.query",
            "user_id": "retriever.user_id",
        }

    def run(self, *, query: str) -> dict[str, list[Document]]:  # noqa: D102
        ...

    def warmup(self) -> None:  # noqa: D102
        ...
