from typing import List

import pytest

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import ListJoiner
from haystack.dataclasses import ChatMessage
from haystack_experimental.core.errors import PipelineBreakException
from haystack_experimental.core.pipeline.pipeline import Pipeline


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
    def test_list_joiner_pipeline(self, list_joiner_pipeline, output_directory, component):

        query = "What is nuclear physics?"
        data = {
            "prompt_builder": {"template_variables": {"query": query}},
            "feedback_prompt_builder": {"template_variables": {"query": query}}
        }

        try:
            _ = list_joiner_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
        except PipelineBreakException as e:
            pass

        all_files = list(output_directory.glob("*"))
        for full_path in all_files:
            f_name = str(full_path).split("/")[-1]
            if str(f_name).startswith(component):
                resume_state = list_joiner_pipeline.load_state(full_path)
                result = list_joiner_pipeline.run(data, resume_state=resume_state)
                assert result
