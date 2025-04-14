import os
from pathlib import Path
from typing import List

import pytest

from haystack.components.converters import OutputAdapter
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack_experimental.core.errors import PipelineBreakpointException
from haystack_experimental.core.pipeline.pipeline import Pipeline

import os
class TestPipelineBreakpoints:

    @pytest.fixture
    def branch_joiner_pipeline(self):
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
        pipe.add_component('fc_llm', OpenAIChatGenerator(model="gpt-4o-mini"))
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
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
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
            # windows paths are not POSIX
            f_name = str(full_path).split("\\")[-1] if os.name == "nt" else str(full_path).split("/")[-1]
            if str(f_name).startswith(component):
                file_found = True
                result = branch_joiner_pipeline.run(data, resume_state_path=full_path)
                assert result['validator']
                break
        if not file_found:
            msg = f"No files found for {component} in {output_directory}."
            raise ValueError(msg)
