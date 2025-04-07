import pytest

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import StringJoiner
from haystack.dataclasses import ChatMessage
from haystack_experimental.core.errors import PipelineBreakException
from haystack_experimental.core.pipeline.pipeline import Pipeline


class TestPipelineBreakpoints:

    @pytest.fixture
    def string_joiner_pipeline(self):
        pipeline = Pipeline()
        pipeline.add_component("prompt_builder_1",
                               ChatPromptBuilder(template=[ChatMessage.from_user("Builder 1: {{query}}")]))
        pipeline.add_component("prompt_builder_2",
                               ChatPromptBuilder(template=[ChatMessage.from_user("Builder 2: {{query}}")]))
        pipeline.add_component("adapter_1", OutputAdapter("{{messages[0].text}}", output_type=str))
        pipeline.add_component("adapter_2", OutputAdapter("{{messages[0].text}}", output_type=str))
        pipeline.add_component("string_joiner", StringJoiner())

        pipeline.connect("prompt_builder_1.prompt", "adapter_1.messages")
        pipeline.connect("prompt_builder_2.prompt", "adapter_2.messages")
        pipeline.connect("adapter_1", "string_joiner.strings")
        pipeline.connect("adapter_2", "string_joiner.strings")

        return pipeline

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    components = [
        "prompt_builder_1",
        "prompt_builder_2",
        "adapter_1",
        "adapter_2",
        "string_joiner"
    ]
    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    def test_list_joiner_pipeline(self, string_joiner_pipeline, output_directory, component):

        string_1 = "What's Natural Language Processing?"
        string_2 = "What is life?"
        data = {"prompt_builder_1": {"query": string_1}, "prompt_builder_2": {"query": string_2}}

        try:
            _ = string_joiner_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
        except PipelineBreakException as e:
            pass

        all_files = list(output_directory.glob("*"))
        for full_path in all_files:
            f_name = str(full_path).split("/")[-1]
            if str(f_name).startswith(component):
                resume_state = string_joiner_pipeline.load_state(full_path)
                result = string_joiner_pipeline.run(data, resume_state=resume_state)
                assert result

