# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from haystack import Document, Pipeline, component
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.components.agents.unified_agent import UnifiedAgent


# Helper to wrap a plain text string in the Jinja2 message block syntax
# required by ChatPromptBuilder when using string templates.
def _user(text: str) -> str:
    return f'{{% message role="user" %}}{text}{{% endmessage %}}'


REPLY_PREFIX = "Reply: "

@component
class MockChatGenerator:
    """Echoes back the last user message."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, list[ChatMessage]]:
        last_user_text = ""
        for m in reversed(messages):
            if m.text is not None:
                last_user_text = m.text
                break
        return {"replies": [ChatMessage.from_assistant(f"{REPLY_PREFIX}{last_user_text}")]}

@component
class MockRetriever:
    """Returns a fixed set of documents."""

    @component.output_types(documents=list[Document])
    def run(self, query: str) -> dict[str, list[Document]]:
        return {
            "documents": [
                Document(content="Paris is the capital of France."),
                Document(content="Berlin is the capital of Germany."),
                Document(content="Madrid is the capital of Spain."),
            ]
        }


def _run_agent(agent: UnifiedAgent, **kwargs) -> dict[str, Any]:
    """Warm-up and run an agent in one call."""
    agent.warm_up()
    return agent.run(**kwargs)


def _make_agent(chat_generator=None, user_prompt=None, **kwargs) -> UnifiedAgent:
    """Shorthand to create a UnifiedAgent with sensible defaults."""
    return UnifiedAgent(
        chat_generator=chat_generator or MockChatGenerator(),
        user_prompt=user_prompt or _user("Hello"),
        **kwargs,
    )


class TestChatPromptBuilderUnification:

    def test_run_static_template(self):
        """Static template should produce a user+assistant pair with correct content."""
        result = _run_agent(_make_agent(user_prompt=_user("Hello, world!")))


        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].role.value == "user"
        assert messages[0].text == "Hello, world!"
        assert messages[1].role.value == "assistant"
        assert messages[1].text == f"{REPLY_PREFIX}Hello, world!"
        assert result["last_message"].text == f"{REPLY_PREFIX}Hello, world!"

    def test_run_static_string_template(self):
        user_prompt = '{% message role="user" %}What is the meaning of life?{% endmessage %}'
        result = _run_agent(_make_agent(user_prompt=user_prompt))

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].role.value == "user"
        assert messages[0].text == "What is the meaning of life?"
        assert messages[1].role.value == "assistant"
        assert messages[1].text == f"{REPLY_PREFIX}What is the meaning of life?"

    def test_run_static_template_multi_message_with_system_prompt(self):
        user_prompt = (
            '{% message role="user" %}First message{% endmessage %}'
            '{% message role="user" %}Second message{% endmessage %}'
        )
        result = _run_agent(_make_agent(user_prompt=user_prompt,
            system_prompt="You are a helpful assistant.",
        ))

        messages = result["messages"]
        assert len(messages) == 4
        assert messages[0].role.value == "system"
        assert messages[0].text == "You are a helpful assistant."
        assert messages[1].role.value == "user"
        assert messages[1].text == "First message"
        assert messages[2].role.value == "user"
        assert messages[2].text == "Second message"
        assert messages[3].role.value == "assistant"
        assert messages[3].text == f"{REPLY_PREFIX}Second message"


    def test_run_string_template_with_multiple_variables(self):
        user_prompt = (
            '{% message role="system" %}You speak {{language}}.{% endmessage %}'
            '{% message role="user" %}{{query}}{% endmessage %}'
        )
        result = _run_agent(
            _make_agent(user_prompt=user_prompt),
            language="English", query="What is AI?",
        )

        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].role.value == "system"
        assert messages[0].text == "You speak English."
        assert messages[1].role.value == "user"
        assert messages[1].text == "What is AI?"
        assert messages[2].role.value == "assistant"
        assert messages[2].text == f"{REPLY_PREFIX}What is AI?"

    def test_run_template_override_at_runtime(self):
        """Passing user_prompt at runtime overrides the init-time template."""
        result = _run_agent(
            _make_agent(user_prompt=_user("Original {{topic}}")),
            user_prompt=_user("Overridden: {{concept}}"),
            concept="deep learning",
        )
        assert result["last_message"].text == f"{REPLY_PREFIX}Overridden: deep learning"

    def test_run_optional_variable_defaults_to_empty(self):
        result = _run_agent(
            _make_agent(user_prompt=_user("Hello {{name}}"))
        )
        assert result["last_message"].text == f"{REPLY_PREFIX}Hello"

    def test_run_rag_pipeline_with_documents_and_query(self):
        """RAG pipeline: a retriever feeds documents into the agent via a Pipeline."""

        user_prompt = (
            '{% message role="user" %}'
            "Answer the question based on the following documents:\n"
            "{% for doc in documents %}"
            "- {{ doc.content }}\n"
            "{% endfor %}\n"
            "Question: {{ query }}"
            "{% endmessage %}"
        )
        agent = _make_agent(user_prompt=user_prompt)

        pipeline = Pipeline()
        pipeline.add_component("retriever", MockRetriever())
        pipeline.add_component("agent", agent)
        pipeline.connect("retriever.documents", "agent.documents")

        query = "What is the capital of France?"
        result = pipeline.run(
            {
                "retriever": { "query": query },
                "agent": { "query": query },
            }
        )

        messages = result["agent"]["messages"]
        assert len(messages) == 2
        assert messages[0].role.value == "user"

        # Verify all document contents from the retriever are rendered in the prompt
        assert "Paris is the capital of France." in messages[0].text
        assert "Berlin is the capital of Germany." in messages[0].text
        assert "Madrid is the capital of Spain." in messages[0].text
        assert query in messages[0].text

        # The mock generator echoes back the rendered user message
        assert messages[1].role.value == "assistant"
        assert messages[1].text == f"{REPLY_PREFIX}{messages[0].text}"
