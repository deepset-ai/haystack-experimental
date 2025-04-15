from typing import Any, Dict, List, Optional, Union
from unittest.mock import patch, MagicMock

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret


class MockOpenAIChatGenerator(OpenAIChatGenerator):
    """
    A mock implementation of OpenAIChatGenerator for testing purposes.
    This component can be used directly in tests to simulate OpenAI responses.
    """

    def __init__(
        self,
        default_content: str = "This is a default response.",
        default_usage: Dict[str, int] = None,
        model_name: str = "gpt-4",
        api_key: str = "test-api-key",
        custom_response_handler: Optional[callable] = None,
    ):
        """
        Initialize the mock generator with customizable parameters.

        Args:
            default_content: The default content to return in the mock response
            default_usage: Token usage statistics for the mock response
            model_name: The model name to use in the mock
            api_key: The API key to use in the mock
            custom_response_handler: Optional function to handle custom response logic
        """
        # Initialize with a patched OpenAI client
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            # Create mock completion object
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(
                    finish_reason="stop",
                    index=0,
                    message=MagicMock(content=default_content)
                )
            ]
            mock_completion.usage = default_usage or {
                "prompt_tokens": 57,
                "completion_tokens": 40,
                "total_tokens": 97
            }
            
            mock_chat_completion_create.return_value = mock_completion
            
            # Call the parent class constructor
            super().__init__(
                model=model_name,
                api_key=Secret.from_token(api_key)
            )
            
            # Store configuration for the run method
            self._default_content = default_content
            self._default_usage = default_usage or {
                "prompt_tokens": 57,
                "completion_tokens": 40,
                "total_tokens": 97
            }
            self._model_name = model_name
            self._custom_response_handler = custom_response_handler

    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[callable] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tools_strict: Optional[bool] = None,
    ) -> Dict[str, Union[List[ChatMessage], Dict[str, Any]]]:
        """
        Mock implementation of the run method.
        
        Args:
            messages: List of chat messages
            streaming_callback: Optional callback for streaming responses
            generation_kwargs: Optional generation parameters
            tools: Optional list of tools
            tools_strict: Optional flag for strict tool usage
            
        Returns:
            Dictionary with replies and metadata
        """
        # Use custom handler if provided, otherwise use default content
        if self._custom_response_handler:
            content = self._custom_response_handler(messages)
        else:
            content = self._default_content
        
        # Return the mock response
        return {
            "replies": [ChatMessage.from_assistant(content)],
            "meta": {
                "model": self._model_name,
                "usage": self._default_usage
            }
        }
        
    async def run_async(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[callable] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tools_strict: Optional[bool] = None,
    ) -> Dict[str, Union[List[ChatMessage], Dict[str, Any]]]:
        """
        Async mock implementation of the run method.
        
        Args:
            messages: List of chat messages
            streaming_callback: Optional callback for streaming responses
            generation_kwargs: Optional generation parameters
            tools: Optional list of tools
            tools_strict: Optional flag for strict tool usage
            
        Returns:
            Dictionary with replies and metadata
        """
        # Use custom handler if provided, otherwise use default content
        if self._custom_response_handler:
            content = self._custom_response_handler(messages)
        else:
            content = self._default_content
        
        # Return the mock response
        return {
            "replies": [ChatMessage.from_assistant(content)],
            "meta": {
                "model": self._model_name,
                "usage": self._default_usage
            }
        }