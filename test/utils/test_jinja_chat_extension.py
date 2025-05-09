# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from jinja2 import TemplateSyntaxError
from jinja2.sandbox import SandboxedEnvironment

from haystack_experimental.dataclasses.chat_message import ImageContent, ToolCall, ToolCallResult
from haystack_experimental.utils.jinja_chat_extension import ChatMessageExtension, for_template


class TestChatMessageExtension:
    @pytest.fixture
    def jinja_env(self) -> SandboxedEnvironment:
        # we use a SandboxedEnvironment here to replicate the conditions of the ChatPromptBuilder component
        env = SandboxedEnvironment(extensions=[ChatMessageExtension])
        env.filters["for_template"] = for_template
        return env

    def test_message_with_name_and_meta(self, jinja_env):
        template = """
        {% message role="user" name="Bob" meta={"language": "en"} %}
        Hello!
        {% endmessage %}
        """
        rendered = jinja_env.from_string(template).render()
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "name": "Bob",
            "content": [{"text": "Hello!"}],
            "meta": {"language": "en"}
        }
        assert output == expected

    def test_message_no_endmessage_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        Hello!
        """
        with pytest.raises(TemplateSyntaxError, match="Jinja was looking for the following tags: 'endmessage'"):
            jinja_env.from_string(template).render()



    def test_system_message(self, jinja_env):
        template = """
        {% message role="system" %}
        You are a helpful assistant.
        {% endmessage %}
        """
        rendered = jinja_env.from_string(template).render()
        output = json.loads(rendered.strip())
        expected = {
            "role": "system",
            "content": [{"text": "You are a helpful assistant."}],
            "name": None,
            "meta": {}
        }
        assert output == expected

    def test_user_message_with_variable(self, jinja_env):
        template = """
        {% message role="user" %}
        Hello, my name is {{name}}!
        {% endmessage %}
        """
        rendered = jinja_env.from_string(template).render(name="Alice")
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "content": [{"text": "Hello, my name is Alice!"}],
            "name": None,
            "meta": {}
        }
        assert output == expected

    def test_assistant_message_with_tool_call(self, jinja_env):
        template = """
        {% message role="assistant" %}
        Let me search for that information.
        {{ tool_call | for_template }}
        {% endmessage %}
        """
        tool_call = ToolCall(
            tool_name="search",
            arguments={"query": "an interesting question"},
            id="search_1"
        )
        rendered = jinja_env.from_string(template).render(tool_call=tool_call)
        output = json.loads(rendered.strip())
        expected = {
            "role": "assistant",
            "content": [
                {"text": "Let me search for that information."},
                {"tool_call": {
                    "tool_name": "search",
                    "arguments": {"query": "an interesting question"},
                    "id": "search_1"
                }}
            ],
            "name": None,
            "meta": {}
        }
        assert output == expected

    def test_tool_message(self, jinja_env):
        template = """
        {% message role="tool" %}
        {{ tool_result | for_template }}
        {% endmessage %}
        """
        tool_call = ToolCall(
            tool_name="search",
            arguments={"query": "test"},
            id="search_1"
        )
        tool_result = ToolCallResult(
            result="Here are the search results",
            origin=tool_call,
            error=False
        )
        rendered = jinja_env.from_string(template).render(tool_result=tool_result)
        output = json.loads(rendered.strip())
        expected = {
            "role": "tool",
            "content": [{
                "tool_call_result": {
                    "result": "Here are the search results",
                    "error": False,
                    "origin": {
                        "tool_name": "search",
                        "arguments": {"query": "test"},
                        "id": "search_1"
                    }
                }
            }],
            "name": None,
            "meta": {}
        }
        assert output == expected

    def test_user_message_with_image(self, jinja_env):
        template = """
        {% message role="user" %}
        Please describe this image:
        {{ image | for_template }}
        {% endmessage %}
        """
        image = ImageContent(base64_image="test_base64", mime_type="image/jpeg")
        rendered = jinja_env.from_string(template).render(image=image)
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "content": [
                {"text": "Please describe this image:"},
                {"image": {
                    "base64_image": "test_base64",
                    "mime_type": "image/jpeg",
                    "detail": None,
                    "meta": {}
                }}
            ],
            "name": None,
            "meta": {}
        }
        assert output == expected


    def test_user_message_with_multiple_images(self, jinja_env):
        template = """
        {% message role="user" %}
        Compare these images:
        {% for img in images %}
        {{ img | for_template }}
        {% endfor %}
        {% endmessage %}
        """
        images = [
            ImageContent(base64_image="test_base64_1", mime_type="image/jpeg"),
            ImageContent(base64_image="test_base64_2", mime_type="image/png")
        ]
        rendered = jinja_env.from_string(template).render(images=images)
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "content": [
                {"text": "Compare these images:"},
                {"image": {
                    "base64_image": "test_base64_1",
                    "mime_type": "image/jpeg",
                    "detail": None,
                    "meta": {}
                }},
                {"image": {
                    "base64_image": "test_base64_2",
                    "mime_type": "image/png",
                    "detail": None,
                    "meta": {}
                }}
            ],
            "name": None,
            "meta": {}
        }
        assert output == expected

    def test_user_message_with_multiple_images_and_interleaved_text(self, jinja_env):
        """
        Tests that messages with multiple images and interleaved text are rendered correctly.
        This format is used by Anthropic models:
        https://docs.anthropic.com/en/docs/build-with-claude/vision#example-multiple-images
        """
        template = """
        {% message role="user" %}
        {% for image in images %}
        Image {{ loop.index }}:
        {{ image | for_template }}
        {% endfor %}
        What's the difference between the two images?
        {% endmessage %}
        """
        image = ImageContent(base64_image="test_base64", mime_type="image/jpeg")
        rendered = jinja_env.from_string(template).render(images=[image, image])
        output = json.loads(rendered.strip())

        expected = {
            "role": "user",
            "content": [{"text": "Image 1:"},
                        {"image": {
                            "base64_image": "test_base64",
                            "mime_type": "image/jpeg",
                            "detail": None,
                            "meta": {}
                        }},
                        {"text": "Image 2:"},
                        {"image": {
                            "base64_image": "test_base64",
                            "mime_type": "image/jpeg",
                            "detail": None,
                            "meta": {}
                        }},
                        {"text": "What's the difference between the two images?"}
            ],
            "name": None,
            "meta": {}
        }
        assert output == expected

    def test_user_message_multiple_lines(self, jinja_env):
        template = """
{% message role="user" %}
What do you think of NLP?
It's an interesting domain, if you ask me.
But my favorite subject is Small Language Models.
{% endmessage %}
        """
        rendered = jinja_env.from_string(template).render()
        output = json.loads(rendered.strip())
        expected = {
            "role": "user",
            "content": [
                {"text": ("What do you think of NLP?\nIt's an interesting domain, if you ask me.\n"
                          "But my favorite subject is Small Language Models."
                          )},
            ],
            "name": None,
            "meta": {}
        }
        assert output == expected

    def test_invalid_role(self, jinja_env):
        template = """
        {% message role="invalid_role" %}
        This should fail
        {% endmessage %}
        """
        with pytest.raises(TemplateSyntaxError, match="Role must be one of"):
            jinja_env.from_string(template).render()

    def test_for_template_filter_with_invalid_type(self):
        with pytest.raises(ValueError, match="Value must be an instance of one of the following types"):
            for_template(123)

    def test_empty_message_content_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        {% endmessage %}
        """
        with pytest.raises(ValueError, match="Message content is empty"):
            jinja_env.from_string(template).render()


    def test_message_with_whitespace_handling(self, jinja_env):
        # the following templates should all be equivalent
        templates = [
            """{% message role="user" %}{{ image | for_template }}{% endmessage %}""",
            """{% message role="user" %}    {{ image | for_template }}    {% endmessage %}""",
            """{% message role="user" %}
            {{ image | for_template }}
            {% endmessage %}""",
            """{% message role="user" %}\t{{ image | for_template }}\t{% endmessage %}"""
        ]
        image = ImageContent(base64_image="test_base64", mime_type="image/jpeg")
        expected = {
            "role": "user",
            "content": [
                {"image": {
                    "base64_image": "test_base64",
                    "mime_type": "image/jpeg",
                    "detail": None,
                    "meta": {}
                }}
            ],
            "name": None,
            "meta": {}
        }
        for template in templates:
            rendered = jinja_env.from_string(template).render(image=image)
            output = json.loads(rendered.strip())
            assert output == expected

    def test_unclosed_content_tag_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        <haystack_content_part>{"type": "text", "text": "Hello"}
        {% endmessage %}
        """
        with pytest.raises(ValueError, match="Found unclosed <haystack_content_part> tag"):
            jinja_env.from_string(template).render()


    def test_invalid_json_in_content_part_raises_error(self, jinja_env):
        template = """
        {% message role="user" %}
        Normal text before.
        <haystack_content_part>{"this is": "invalid" json}</haystack_content_part>
        <haystack_content_part>not even trying to be json</haystack_content_part>
        <haystack_content_part>{]</haystack_content_part>
        Normal text after.
        {% endmessage %}
        """
        with pytest.raises(json.JSONDecodeError):
            jinja_env.from_string(template).render()


