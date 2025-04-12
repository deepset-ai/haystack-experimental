import os

import pytest
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_experimental.core.pipeline.pipeline import Pipeline
from haystack_experimental.core.errors import PipelineBreakpointException
from pydantic import BaseModel
from typing import List, Optional
import json

import pydantic
from pydantic import ValidationError
from colorama import Fore
from haystack import component

import os

# Define the component input parameters
@component
class OutputValidator:
    def __init__(self, pydantic_model: pydantic.BaseModel):
        self.pydantic_model = pydantic_model
        self.iteration_counter = 0

    # Define the component output
    @component.output_types(valid_replies=List[str], invalid_replies=Optional[List[str]], error_message=Optional[str])
    def run(self, replies: List[ChatMessage]):

        self.iteration_counter += 1

        ## Try to parse the LLM's reply ##
        # If the LLM's reply is a valid object, return `"valid_replies"`
        try:
            output_dict = json.loads(replies[0].text)
            self.pydantic_model.model_validate(output_dict)
            print(
                Fore.GREEN
                + f"OutputValidator at Iteration {self.iteration_counter}: Valid JSON from LLM - No need for looping: {replies[0]}"
            )
            return {"valid_replies": replies}

        # If the LLM's reply is corrupted or not valid, return "invalid_replies" and the "error_message" for LLM to try again
        except (ValueError, ValidationError) as e:
            print(
                Fore.RED
                + f"OutputValidator at Iteration {self.iteration_counter}: Invalid JSON from LLM - Let's try again.\n"
                f"Output from LLM:\n {replies[0]} \n"
                f"Error from OutputValidator: {e}"
            )
            return {"invalid_replies": replies, "error_message": str(e)}


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]


class TestPipelineBreakpointsLoops:
    """
    This class contains tests for pipelines with validation loops and breakpoints.
    """

    @pytest.fixture
    def validation_loop_pipeline(self):
        """Create a pipeline with validation loops for testing."""
        prompt_template = [
            ChatMessage.from_user(
                """
                Create a JSON object from the information present in this passage: {{passage}}.
                Only use information that is present in the passage. Follow this JSON schema, but only return the actual instances without any additional schema definition:
                {{schema}}
                Make sure your response is a dict and not a list.
                {% if invalid_replies and error_message %}
                  You already created the following output in a previous attempt: {{invalid_replies}}
                  However, this doesn't comply with the format requirements from above and triggered this Python exception: {{error_message}}
                  Correct the output and try again. Just return the corrected output without any extra explanations.
                {% endif %}
                """
            )
        ]

        pipeline = Pipeline(max_runs_per_component=5)
        pipeline.add_component(instance=ChatPromptBuilder(template=prompt_template), name="prompt_builder")
        pipeline.add_component(instance=OpenAIChatGenerator(), name="llm")
        pipeline.add_component(instance=OutputValidator(pydantic_model=CitiesData), name="output_validator")

        # Connect components
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        pipeline.connect("llm.replies", "output_validator")
        pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")
        pipeline.connect("output_validator.error_message", "prompt_builder.error_message")

        return pipeline

    @pytest.fixture(scope="session")
    def output_directory(self, tmp_path_factory):
        return tmp_path_factory.mktemp("output_files")

    @pytest.fixture
    def test_data(self):
        json_schema = {
            "cities": [
                {
                    "name": "Berlin",
                    "country": "Germany",
                    "population": 3850809
                },
                {
                    "name": "Paris",
                    "country": "France",
                    "population": 2161000
                },
                {
                    "name": "Lisbon",
                    "country": "Portugal",
                    "population": 504718
                }
            ]
        }

        passage = "Berlin is the capital of Germany. It has a population of 3,850,809. Paris, France's capital, has 2.161 million residents. Lisbon is the capital and the largest city of Portugal with the population of 504,718."

        return {"schema": json_schema, "passage": passage}

    components = [
        "prompt_builder",
        "llm",
        "output_validator"
    ]
    @pytest.mark.parametrize("component", components)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_pipeline_breakpoints_validation_loop(self, validation_loop_pipeline, output_directory, test_data, component):
        """
        Test that a pipeline with validation loops can be executed with breakpoints at each component.
        """
        data = {
            "prompt_builder": {
                "passage": test_data["passage"],
                "schema": test_data["schema"]
            }
        }

        try:
            _ = validation_loop_pipeline.run(data, breakpoints={(component, 0)}, debug_path=str(output_directory))
        except PipelineBreakpointException:
            pass

        all_files = list(output_directory.glob("*"))
        file_found = False
        for full_path in all_files:
            f_name = str(full_path).split("/")[-1]
            if str(f_name).startswith(component):
                file_found = True
                result = validation_loop_pipeline.run(data={}, resume_state_path=full_path)
                # Verify the result contains valid output
                if "output_validator" in result and "valid_replies" in result["output_validator"]:
                    valid_reply = result["output_validator"]["valid_replies"][0].text
                    valid_json = json.loads(valid_reply)
                    assert isinstance(valid_json, dict)
                    assert "cities" in valid_json
                    cities_data = CitiesData.model_validate(valid_json)
                    assert len(cities_data.cities) == 3
        if not file_found:
            raise ValueError("No files found for {component} in {output_directory}.")