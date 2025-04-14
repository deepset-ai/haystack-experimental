import os
from pathlib import Path

import pytest

from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.joiners import AnswerJoiner
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage
from haystack_experimental.core.errors import PipelineBreakpointException
from haystack_experimental.core.pipeline.pipeline import Pipeline

import os

class TestPipelineBreakpoints:

    @pytest.fixture
    def answer_join_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("gpt-4o", OpenAIChatGenerator(model="gpt-4o"))
        pipeline.add_component("gpt-3", OpenAIChatGenerator(model="gpt-3.5-turbo"))
        pipeline.add_component("answer_builder_a", AnswerBuilder())
        pipeline.add_component("answer_builder_b", AnswerBuilder())
        pipeline.add_component("answer_joiner", AnswerJoiner())
        pipeline.connect("gpt-4o.replies", "answer_builder_a")
        pipeline.connect("gpt-3.replies", "answer_builder_b")
        pipeline.connect("answer_builder_a.answers", "answer_joiner")
        pipeline.connect("answer_builder_b.answers", "answer_joiner")

        return pipeline

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    components = [
        "gpt-4o",
        "gpt-3",
        "answer_builder_a",
        "answer_builder_b",
        "answer_joiner",
    ]
    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_pipeline_breakpoints_answer_joiner(self, answer_join_pipeline, output_directory, component):

        query = "What's Natural Language Processing?"
        messages = [
            ChatMessage.from_system("You are a helpful, respectful and honest assistant. Be super concise."),
            ChatMessage.from_user(query)
        ]
        data = {
            "gpt-4o": {"messages": messages},
            "gpt-3": {"messages": messages},
            "answer_builder_a": {"query": query},
            "answer_builder_b": {"query": query}
        }

        try:
            _ = answer_join_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
        except PipelineBreakpointException as e:
            pass

        all_files = list(output_directory.glob("*"))

        print("\nAll files in output directory:")
        for f_name in all_files:
            print(f_name)

        file_found = False
        for full_path in all_files:
            f_name = str(full_path).split("/")[-1]
            if str(f_name).startswith(component):
                file_found = True
                result = answer_join_pipeline.run(data, breakpoints=None, resume_state_path=full_path)
                assert result['answer_joiner']
                break
        if not file_found:
            msg = f"No files found for {component} in {output_directory}."
            raise ValueError(msg)
