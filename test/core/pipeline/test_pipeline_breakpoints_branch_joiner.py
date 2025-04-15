import os
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

import pytest

from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.utils.auth import Secret
from haystack_experimental.core.errors import PipelineBreakpointException
from haystack_experimental.core.pipeline.pipeline import Pipeline

import os
class TestPipelineBreakpoints:

    @pytest.fixture
    def mock_openai_chat_generator(self):
        """
        Creates a mock for the OpenAIChatGenerator.
        """
        with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
            # Create mock completion objects
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(
                    finish_reason="stop",
                    index=0,
                    message=MagicMock(content='{"first_name": "Peter", "last_name": "Parker", "nationality": "American"}')
                )
            ]
            mock_completion.usage = {
                "prompt_tokens": 57,
                "completion_tokens": 40,
                "total_tokens": 97
            }
            
            mock_chat_completion_create.return_value = mock_completion
            
            # Create a mock for the OpenAIChatGenerator
            def create_mock_generator(model_name):
                generator = OpenAIChatGenerator(model=model_name, api_key=Secret.from_token("test-api-key"))
                
                # Mock the run method
                def mock_run(messages, streaming_callback=None, generation_kwargs=None, tools=None, tools_strict=None):
                    content = '{"first_name": "Peter", "last_name": "Parker", "nationality": "American"}'
                    
                    return {
                        "replies": [ChatMessage.from_assistant(content)],
                        "meta": {"model": model_name, "usage": {"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97}}
                    }
                
                # Replace the run method with our mock
                generator.run = mock_run
                
                return generator
            
            yield create_mock_generator

    @pytest.fixture
    def branch_joiner_pipeline(self, mock_openai_chat_generator):
        person_schema = {
            "type": "object",
            "properties": {
                "first_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                "last_name": {"type": "string", "pattern": "^[A-Z][a-z]+$"},
                "nationality": {"type": "string", "enum": ["Italian", "Portuguese", "American"]},
            },
            "required": ["first_name", "last_name", "nationality"]
        }

        pipe = Pipeline()
        pipe.add_component('joiner', BranchJoiner(List[ChatMessage]))
        pipe.add_component('fc_llm', mock_openai_chat_generator("gpt-4o-mini"))
        pipe.add_component('validator', JsonSchemaValidator(json_schema=person_schema))
        pipe.add_component('adapter', OutputAdapter("{{chat_message}}", List[ChatMessage], unsafe=True))

        pipe.connect("adapter", "joiner")
        pipe.connect("joiner", "fc_llm")
        pipe.connect("fc_llm.replies", "validator.messages")
        pipe.connect("validator.validation_error", "joiner")

        return pipe

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    components = [
        "joiner",
        "fc_llm",
        "validator",
        "adapter",
    ]
    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    def test_pipeline_breakpoints_branch_joiner(self, branch_joiner_pipeline, output_directory, component):

        data = {
            "fc_llm": {"generation_kwargs": {"response_format": {"type": "json_object"}}},
            "adapter": {"chat_message": [ChatMessage.from_user("Create JSON from Peter Parker")]}
        }

        try:
            _ = branch_joiner_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
        except PipelineBreakpointException as e:
            pass

        all_files = list(output_directory.glob("*"))
        file_found = False
        for full_path in all_files:
            f_name = Path(full_path).name
            if str(f_name).startswith(component):
                file_found = True
                resume_state = Pipeline.load_state(full_path)
                result = branch_joiner_pipeline.run(data, resume_state=resume_state)
                assert result['validator']
                break
        if not file_found:
            msg = f"No files found for {component} in {output_directory}."
            raise ValueError(msg)

