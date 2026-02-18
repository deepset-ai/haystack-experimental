# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from haystack import Pipeline, component
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import Tool, create_tool_from_function

from haystack_experimental.super_components.agents.integrated_agent import IntegratedAgent


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
        return {"replies": [ChatMessage.from_assistant(f"Reply: {last_user_text}")]}


@component
class MockAllMessagesGenerator:
    """Summarizes all received messages so tests can inspect the full conversation."""

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, list[ChatMessage]]:
        parts = [f"{m.role.value}:{m.text}" for m in messages]
        return {"replies": [ChatMessage.from_assistant("Received: " + " | ".join(parts))]}


def _make_tool_chat_generator(call_limit: int = 1):
    """
    Factory returning a mock chat generator that issues a tool call for the
    first *call_limit* invocations, then returns a plain text reply.
    """
    counter = {"n": 0}

    @component
    class _MockToolChatGenerator:
        @component.output_types(replies=list[ChatMessage])
        def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, list[ChatMessage]]:
            counter["n"] += 1
            if counter["n"] <= call_limit:
                return {
                    "replies": [
                        ChatMessage.from_assistant(
                            tool_calls=[ToolCall(tool_name="addition_tool", arguments={"a": 2, "b": 3})]
                        )
                    ]
                }
            return {"replies": [ChatMessage.from_assistant("The result is 5.")]}

    return _MockToolChatGenerator()


def addition_tool(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b


@pytest.fixture
def tool() -> Tool:
    return create_tool_from_function(
        function=addition_tool,
        name="addition_tool",
        description="Adds two numbers",
    )


def _run_agent(agent: IntegratedAgent, **kwargs) -> dict[str, Any]:
    """Warm-up and run an agent in one call."""
    agent.warm_up()
    return agent.run(**kwargs)


def _make_agent(chat_generator=None, template=None, **kwargs) -> IntegratedAgent:
    """Shorthand to create an IntegratedAgent with sensible defaults."""
    return IntegratedAgent(
        chat_generator=chat_generator or MockChatGenerator(),
        template=template or [ChatMessage.from_user("Hello")],
        **kwargs,
    )


class TestRunNoTemplateVariables:

    def test_run_static_template(self):
        """Static template should produce a user+assistant pair with correct content."""
        result = _run_agent(_make_agent(template=[ChatMessage.from_user("Hello, world!")]))

        assert result["last_message"].text == "Reply: Hello, world!"

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].role.value == "user"
        assert messages[0].text == "Hello, world!"
        assert messages[1].role.value == "assistant"
        assert set(result.keys()) == {"messages", "last_message"}

    def test_run_static_string_template(self):
        tpl = '{% message role="user" %}What is the meaning of life?{% endmessage %}'
        result = _run_agent(_make_agent(template=tpl))

        assert result["last_message"].text == "Reply: What is the meaning of life?"

    def test_run_static_template_with_system_prompt(self):
        result = _run_agent(_make_agent(
            chat_generator=MockAllMessagesGenerator(),
            template=[ChatMessage.from_user("Hello")],
            system_prompt="You are a helpful assistant.",
        ))

        messages = result["messages"]
        assert messages[0].role.value == "system"
        assert messages[0].text == "You are a helpful assistant."
        assert messages[1].role.value == "user"
        assert messages[1].text == "Hello"

    def test_run_static_template_multi_message(self):
        result = _run_agent(_make_agent(template=[
            ChatMessage.from_user("First message"),
            ChatMessage.from_user("Second message"),
        ]))

        assert result["last_message"].text == "Reply: Second message"
        assert len(result["messages"]) >= 3  # 2 user + 1 assistant


class TestRunComplexTemplateVariables:

    def test_run_single_variable(self):
        result = _run_agent(
            _make_agent(template=[ChatMessage.from_user("Tell me about {{topic}}")]),
            topic="Haystack",
        )
        assert result["last_message"].text == "Reply: Tell me about Haystack"

    def test_run_multiple_variables(self):
        result = _run_agent(
            _make_agent(template=[ChatMessage.from_user("Name: {{name}}, Age: {{age}}, City: {{city}}")]),
            name="Alice", age="30", city="Berlin",
        )
        text = result["last_message"].text
        assert "Alice" in text and "30" in text and "Berlin" in text

    def test_run_string_template_with_multiple_variables(self):
        tpl = (
            '{% message role="system" %}You speak {{language}}.{% endmessage %}'
            '{% message role="user" %}{{question}}{% endmessage %}'
        )
        result = _run_agent(
            _make_agent(chat_generator=MockChatGenerator(), template=tpl),
            language="English", question="What is AI?",
        )

        assert result["last_message"].text == "Reply: What is AI?"
        messages = result["messages"]
        assert len(messages) == 3
        assert messages[0].role.value == "system"
        assert messages[0].text == "You speak English."
        assert messages[1].role.value == "user"
        assert messages[1].text == "What is AI?"

    def test_run_template_override_at_runtime(self):
        result = _run_agent(
            _make_agent(template=[ChatMessage.from_user("Original {{topic}}")]),
            template=[ChatMessage.from_user("Overridden: {{concept}}")],
            template_variables={"concept": "deep learning"},
        )
        assert result["last_message"].text == "Reply: Overridden: deep learning"

    def test_run_template_variables_take_precedence_over_kwargs(self):
        result = _run_agent(
            _make_agent(template=[ChatMessage.from_user("Hello {{name}}")]),
            name="FromKwarg",
            template_variables={"name": "FromTemplateVars"},
        )
        assert "FromTemplateVars" in result["last_message"].text

    def test_run_optional_variable_defaults_to_empty(self):
        result = _run_agent(
            _make_agent(template=[ChatMessage.from_user("Hello {{name}}")])
        )
        assert result["last_message"].text == "Reply: Hello "

    def test_run_required_variable_missing_raises(self):
        agent = _make_agent(
            template=[ChatMessage.from_user("Hello {{name}}")],
            required_variables=["name"],
        )
        agent.warm_up()
        with pytest.raises(ValueError, match="Missing mandatory input"):
            agent.run()

    @pytest.mark.parametrize("value,label", [
        ("This is a very long passage. " * 50, "long_text"),
        ("Hello <world> & 'friends' \"everyone\" {braces}", "special_chars"),
    ], ids=["long_text", "special_chars"])
    def test_run_variable_with_edge_case_values(self, value, label):
        result = _run_agent(
            _make_agent(template=[ChatMessage.from_user("Input: {{data}}")]),
            data=value,
        )
        assert value in result["last_message"].text

    def test_run_system_prompt_with_string_template(self):
        tpl = '{% message role="user" %}{{question}}{% endmessage %}'
        result = _run_agent(_make_agent(
            chat_generator=MockAllMessagesGenerator(),
            template=tpl,
            system_prompt="You are a bot.",
        ), question="Hi")

        messages = result["messages"]
        assert messages[0].role.value == "system"
        assert messages[0].text == "You are a bot."

    def test_run_mixed_required_and_optional_variables(self):
        agent = _make_agent(
            template=[ChatMessage.from_user("{{greeting}} {{name}}")],
            required_variables=["greeting"],
        )
        agent.warm_up()

        # Providing the required var but not the optional one should succeed
        result = agent.run(greeting="Hi")
        assert "Hi" in result["last_message"].text

        # Missing the required var should raise
        with pytest.raises(ValueError, match="Missing mandatory input"):
            agent.run(name="Alice")


class TestRunNoTools:

    def test_run_no_tools_basic(self):
        """Single-step run: correct reply, 2 messages, no tool calls, expected keys."""
        result = _run_agent(
            _make_agent(template=[ChatMessage.from_user("Hello {{name}}")]),
            name="World",
        )

        assert result["last_message"].text == "Reply: Hello World"
        assert set(result.keys()) == {"messages", "last_message"}

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0].role.value == "user"
        assert messages[1].role.value == "assistant"

        last = result["last_message"]
        assert last.tool_calls is None or len(last.tool_calls) == 0

    def test_run_no_tools_embedded_in_pipeline(self):
        agent = _make_agent(template=[ChatMessage.from_user("Tell me about {{topic}}")])

        pipe = Pipeline()
        pipe.add_component("agent", agent)
        pipe.warm_up()

        result = pipe.run(data={"agent": {"topic": "Haystack"}})
        assert result["agent"]["last_message"].text == "Reply: Tell me about Haystack"
        assert "messages" in result["agent"]

    def test_run_no_tools_with_system_prompt(self):
        result = _run_agent(_make_agent(
            chat_generator=MockAllMessagesGenerator(),
            template=[ChatMessage.from_user("Hey there")],
            system_prompt="Be brief.",
        ))

        messages = result["messages"]
        assert messages[0].role.value == "system"
        assert messages[0].text == "Be brief."
        assert messages[1].role.value == "user"
        assert messages[1].text == "Hey there"
        assert messages[2].role.value == "assistant"


class TestRunWithTools:

    def test_run_with_tools_message_flow(self, tool):
        """Full message sequence: user → assistant(tool_call) → tool → assistant(text)."""
        result = _run_agent(
            _make_agent(
                chat_generator=_make_tool_chat_generator(call_limit=1),
                template=[ChatMessage.from_user("{{query}}")],
                tools=[tool],
            ),
            query="What is 2 + 3?",
        )

        assert result["last_message"].text == "The result is 5."
        assert set(result.keys()) == {"messages", "last_message"}

        messages = result["messages"]
        assert len(messages) == 4
        assert messages[0].role.value == "user"
        assert messages[1].role.value == "assistant"
        assert messages[1].tool_calls is not None
        assert len(messages[1].tool_calls) == 1
        assert messages[1].tool_calls[0].tool_name == "addition_tool"
        assert messages[2].role.value == "tool"
        assert messages[3].role.value == "assistant"
        assert messages[3].text == "The result is 5."

    def test_run_with_tools_exit_condition(self, tool):
        """exit_conditions with a tool name should exit after that tool is called."""
        result = _run_agent(
            _make_agent(
                chat_generator=_make_tool_chat_generator(call_limit=10),
                template=[ChatMessage.from_user("{{query}}")],
                tools=[tool],
                exit_conditions=["addition_tool"],
            ),
            query="Add 2 + 3",
        )

        assert result["last_message"].tool_call_results is not None

    def test_run_with_tools_max_steps_limits_execution(self, tool):
        @component
        class AlwaysToolCallGenerator:
            @component.output_types(replies=list[ChatMessage])
            def run(self, messages: list[ChatMessage], tools: Any = None) -> dict[str, list[ChatMessage]]:
                return {
                    "replies": [
                        ChatMessage.from_assistant(
                            tool_calls=[ToolCall(tool_name="addition_tool", arguments={"a": 1, "b": 1})]
                        )
                    ]
                }

        result = _run_agent(
            _make_agent(
                chat_generator=AlwaysToolCallGenerator(),
                template=[ChatMessage.from_user("{{q}}")],
                tools=[tool],
                max_agent_steps=3,
            ),
            q="go",
        )
        assert "messages" in result

    def test_run_with_tools_embedded_in_pipeline(self, tool):
        agent = _make_agent(
            chat_generator=_make_tool_chat_generator(call_limit=1),
            template=[ChatMessage.from_user("{{query}}")],
            tools=[tool],
        )

        pipe = Pipeline()
        pipe.add_component("agent", agent)
        pipe.warm_up()

        result = pipe.run(data={"agent": {"query": "What is 2+3?"}})
        assert result["agent"]["last_message"].text == "The result is 5."

    def test_run_with_tools_and_complex_template(self, tool):
        tpl = (
            '{% message role="system" %}You are a {{role}}.{% endmessage %}'
            '{% message role="user" %}{{query}}{% endmessage %}'
        )
        result = _run_agent(
            _make_agent(
                chat_generator=_make_tool_chat_generator(call_limit=1),
                template=tpl,
                tools=[tool],
            ),
            role="calculator", query="What is 2 + 3?",
        )

        assert result["last_message"].text == "The result is 5."
        # system + user + assistant(tool_call) + tool(result) + assistant(text)
        assert len(result["messages"]) == 5
        assert result["messages"][0].role.value == "system"
        assert result["messages"][0].text == "You are a calculator."

    def test_raise_on_tool_invocation_failure_flag(self):
        def failing_tool(x: int) -> int:
            """A tool that always fails."""
            raise RuntimeError("Tool failed!")

        failing = create_tool_from_function(failing_tool, name="failing_tool", description="Always fails")
        agent = _make_agent(
            template=[ChatMessage.from_user("{{q}}")],
            tools=[failing],
            raise_on_tool_invocation_failure=True,
        )

        inner_agent = agent.pipeline.graph.nodes["agent"]["instance"]
        assert inner_agent.raise_on_tool_invocation_failure is True

    def test_run_with_multiple_tool_calls(self, tool):
        """Three sequential tool-call cycles before a text reply."""
        result = _run_agent(
            _make_agent(
                chat_generator=_make_tool_chat_generator(call_limit=3),
                template=[ChatMessage.from_user("{{q}}")],
                tools=[tool],
            ),
            q="Do some math",
        )

        assert result["last_message"].text == "The result is 5."
        # user + 3*(assistant + tool) + assistant(text) = 8
        assert len(result["messages"]) == 8

    def test_run_with_tools_and_system_prompt(self, tool):
        result = _run_agent(
            _make_agent(
                chat_generator=_make_tool_chat_generator(call_limit=1),
                template=[ChatMessage.from_user("{{query}}")],
                tools=[tool],
                system_prompt="You are a calculator bot.",
            ),
            query="What is 2 + 3?",
        )

        assert result["last_message"].text == "The result is 5."
        # system + user + assistant(tool_call) + tool(result) + assistant(text) = 5
        assert result["messages"][0].role.value == "system"
        assert result["messages"][0].text == "You are a calculator bot."
