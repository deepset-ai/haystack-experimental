import pytest
import os
from haystack.dataclasses.chat_message import ChatMessage
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator


@pytest.fixture
def messages():
    memory_instruction = [ChatMessage.from_system(

            " You are a helpful assistant that can give short and concise answers. When messages start with `[MEMORY]`, treat them as long-term context and use them to guide the response if relevant."
        )]
    user_message=[ChatMessage.from_user(
                "I am building a travel chatbot for my startup Xavier. Based on my preferred programming language, suggest me a database for the travel chatbot. Keep it brief."
            )]

    memories = [ChatMessage.from_system("""
    Here are the relevant memories for the user's query:
    - MEMORY #1: The user is a software engineer and likes building applications in Python.
    - MEMORY #2: The user's most projects are related to NLP and LLM agents.
    - MEMORY #3: The user finds it easier to use the Haystack framework to build projects.
    """)]
    messages = memory_instruction + user_message + memories
    return messages

class TestMemory:

    @pytest.mark.integration
    def test_memory_with_claude(self, messages):
        chat_generator = AmazonBedrockChatGenerator(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
        ans = chat_generator.run(messages=messages)
        assert "python" in ans["replies"][0].text.lower()

    @pytest.mark.integration
    def test_memory_with_amazon_nova(self, messages):
        chat_generator = AmazonBedrockChatGenerator(model="eu.amazon.nova-lite-v1:0")
        ans = chat_generator.run(messages=messages)
        assert "python" in ans["replies"][0].text.lower()

    @pytest.mark.integration
    def test_memory_with_meta_llama(self, messages):
        chat_generator = AmazonBedrockChatGenerator(model="eu.meta.llama3-2-3b-instruct-v1:0")
        ans = chat_generator.run(messages=messages)
        assert "python" in ans["replies"][0].text.lower()

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set")
    def test_memory_with_openai(self, messages):
        chat_generator = OpenAIChatGenerator(model="gpt-4o-mini")
        ans = chat_generator.run(messages=messages)
        assert "python" in ans["replies"][0].text.lower()

    @pytest.mark.integration
    @pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY is not set")
    def test_memory_with_google_gemini(self, messages):
        chat_generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash-lite")
        ans = chat_generator.run(messages=messages)
        assert "python" in ans["replies"][0].text.lower()
