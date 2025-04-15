import pytest
from haystack.dataclasses import ChatMessage
from test.mocks.openai_chat_generator import MockOpenAIChatGenerator


def test_basic_usage():
    """
    Basic example of using the MockOpenAIChatGenerator with default settings.
    """
    # Create a mock generator with default settings
    mock_generator = MockOpenAIChatGenerator()
    
    # Create a list of messages to pass to the run method
    messages = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What is the capital of France?")
    ]
    
    # Run the generator with the messages
    result = mock_generator.run(messages=messages)
    
    # Check the result
    assert len(result["replies"]) == 1
    assert result["replies"][0].content == "This is a default response."
    assert result["meta"]["model"] == "gpt-4"
    assert "usage" in result["meta"]


def test_custom_response():
    """
    Example of using the MockOpenAIChatGenerator with a custom response.
    """
    # Create a mock generator with a custom response
    mock_generator = MockOpenAIChatGenerator(
        default_content="The capital of France is Paris.",
        model_name="gpt-3.5-turbo"
    )
    
    # Create a list of messages
    messages = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What is the capital of France?")
    ]
    
    # Run the generator
    result = mock_generator.run(messages=messages)
    
    # Check the result
    assert result["replies"][0].content == "The capital of France is Paris."
    assert result["meta"]["model"] == "gpt-3.5-turbo"


def test_custom_response_handler():
    """
    Example of using the MockOpenAIChatGenerator with a custom response handler.
    """
    # Define a custom response handler
    def custom_handler(messages):
        # Extract the user's question from the last message
        user_question = messages[-1].content
        # Return a custom response based on the question
        if "capital" in user_question.lower():
            return "The capital of France is Paris."
        elif "weather" in user_question.lower():
            return "The weather in Paris is currently sunny."
        else:
            return "I don't have information about that."
    
    # Create a mock generator with the custom handler
    mock_generator = MockOpenAIChatGenerator(
        custom_response_handler=custom_handler
    )
    
    # Test with different questions
    capital_question = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What is the capital of France?")
    ]
    weather_question = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("What's the weather like in Paris?")
    ]
    other_question = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("Tell me about quantum physics.")
    ]
    
    # Run the generator with different questions
    capital_result = mock_generator.run(messages=capital_question)
    weather_result = mock_generator.run(messages=weather_question)
    other_result = mock_generator.run(messages=other_question)
    
    # Check the results
    assert "The capital of France is Paris." in capital_result["replies"][0].content
    assert "The weather in Paris is currently sunny." in weather_result["replies"][0].content
    assert "I don't have information about that." in other_result["replies"][0].content


@pytest.mark.asyncio
async def test_async_usage():
    """
    Example of using the MockOpenAIChatGenerator in an async context.
    """
    # Create a mock generator
    mock_generator = MockOpenAIChatGenerator(
        default_content="This is an async response."
    )
    
    # Create a list of messages
    messages = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("Hello, can you help me?")
    ]
    
    # Run the generator asynchronously
    result = await mock_generator.run_async(messages=messages)
    
    # Check the result
    assert result["replies"][0].content == "This is an async response."


def test_in_pipeline():
    """
    Example of using the MockOpenAIChatGenerator in a pipeline.
    """
    from haystack.core.pipeline import Pipeline
    
    # Create a mock generator
    mock_generator = MockOpenAIChatGenerator(
        default_content="This is a pipeline response."
    )
    
    # Create a simple pipeline with the mock generator
    pipeline = Pipeline()
    pipeline.add_component("chat_generator", mock_generator)
    
    # Create a list of messages
    messages = [
        ChatMessage.from_system("You are a helpful assistant."),
        ChatMessage.from_user("Hello, can you help me?")
    ]
    
    # Run the pipeline
    result = pipeline.run({"chat_generator": {"messages": messages}})
    
    # Check the result
    assert result["chat_generator"]["replies"][0].content == "This is a pipeline response." 