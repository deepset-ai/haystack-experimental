import os
from typing import List

import pytest

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import ListJoiner
from haystack.dataclasses import ChatMessage
from haystack_experimental.core.errors import PipelineBreakpointException
from haystack_experimental.core.pipeline.pipeline import Pipeline

import os
class TestPipelineBreakpoints:

    @pytest.fixture
    def list_joiner_pipeline(self):
        user_message = [ChatMessage.from_user("Give a brief answer the following question: {{query}}")]

        feedback_prompt = """
            You are given a question and an answer.
            Your task is to provide a score and a brief feedback on the answer.
            Question: {{query}}
            Answer: {{response}}
            """

        feedback_message = [ChatMessage.from_system(feedback_prompt)]

        prompt_builder = ChatPromptBuilder(template=user_message)
        feedback_prompt_builder = ChatPromptBuilder(template=feedback_message)
        llm = OpenAIChatGenerator(model="gpt-4o-mini")
        feedback_llm = OpenAIChatGenerator(model="gpt-4o-mini")

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.add_component("llm", llm)
        pipe.add_component("feedback_prompt_builder", feedback_prompt_builder)
        pipe.add_component("feedback_llm", feedback_llm)
        pipe.add_component("list_joiner", ListJoiner(List[ChatMessage]))

        pipe.connect("prompt_builder.prompt", "llm.messages")
        pipe.connect("prompt_builder.prompt", "list_joiner")
        pipe.connect("llm.replies", "list_joiner")
        pipe.connect("llm.replies", "feedback_prompt_builder.response")
        pipe.connect("feedback_prompt_builder.prompt", "feedback_llm.messages")
        pipe.connect("feedback_llm.replies", "list_joiner")

        return pipe

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    components = [
        "prompt_builder",
        "llm",
        "feedback_prompt_builder",
        "feedback_llm",
        "list_joiner"
    ]
    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_list_joiner_pipeline(self, list_joiner_pipeline, output_directory, component):

        query = "What is nuclear physics?"
        data = {
            "prompt_builder": {"template_variables": {"query": query}},
            "feedback_prompt_builder": {"template_variables": {"query": query}}
        }

        try:
            _ = list_joiner_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
        except PipelineBreakpointException as e:
            pass

        all_files = list(output_directory.glob("*"))
        file_found = False
        for full_path in all_files:
            if os.name == "nt":  # windows paths are not POSIX
                f_name = str(full_path).split("\\")[-1]
            else:
                f_name = str(full_path).split("/")[-1]

            if str(f_name).startswith(component):
                file_found = True
                result = list_joiner_pipeline.run(data, resume_state_path=full_path)
                assert result['list_joiner']
        if not file_found:
            msg = f"No files found for {component} in {output_directory}."
            raise ValueError(msg)
