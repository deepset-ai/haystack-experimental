import pytest

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters import OutputAdapter
from haystack.components.joiners import StringJoiner
from haystack.dataclasses import ChatMessage
from haystack_experimental.core.errors import BreakpointException
from haystack_experimental.core.pipeline.pipeline import Pipeline
from haystack_experimental.dataclasses.breakpoints import Breakpoint
from test.conftest import load_and_resume_pipeline_state


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
        Breakpoint("prompt_builder_1", 0),
        Breakpoint("prompt_builder_2", 0),
        Breakpoint("adapter_1", 0),
        Breakpoint("adapter_2", 0),
        Breakpoint("string_joiner", 0)
    ]
    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    def test_string_joiner_pipeline(self, string_joiner_pipeline, output_directory, component):
        string_1 = "What's Natural Language Processing?"
        string_2 = "What is life?"
        data = {"prompt_builder_1": {"query": string_1}, "prompt_builder_2": {"query": string_2}}

        try:
            _ = string_joiner_pipeline.run(data, break_point=component, debug_path=str(output_directory))
        except BreakpointException as e:
            pass

        result = load_and_resume_pipeline_state(string_joiner_pipeline, output_directory, component.component_name, data)
        assert result['string_joiner']
