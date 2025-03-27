from typing import List
from pydantic import BaseModel
import argparse


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]


import json
import random
import pydantic
from pydantic import ValidationError
from typing import Optional, List
from colorama import Fore
from haystack import component
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.builders import ChatPromptBuilder
from haystack_experimental.core.pipeline.pipeline import Pipeline


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
            self.pydantic_model.parse_obj(output_dict)
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

def create_pipeline():
    """
    Creates and returns the pipeline with validation loop components.
    """
    output_validator = OutputValidator(pydantic_model=CitiesData)

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
    prompt_builder = ChatPromptBuilder(template=prompt_template)
    chat_generator = OpenAIChatGenerator()

    pipeline = Pipeline(max_runs_per_component=5)

    # Add components to pipeline
    pipeline.add_component(instance=prompt_builder, name="prompt_builder")
    pipeline.add_component(instance=chat_generator, name="llm")
    pipeline.add_component(instance=output_validator, name="output_validator")

    # Connect components
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    pipeline.connect("llm.replies", "output_validator")
    pipeline.connect("output_validator.invalid_replies", "prompt_builder.invalid_replies")
    pipeline.connect("output_validator.error_message", "prompt_builder.error_message")

    return pipeline

def breakpoint():
    """
    Runs the pipeline with a breakpoint before the LLM.
    """
    pipeline = create_pipeline()

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

    result = pipeline.run(
        {"prompt_builder": {"passage": passage, "schema": json_schema}},
        breakpoints={("llm", 0)}
    )
    return result

def resume(resume_state):
    """
    Resumes the pipeline from a saved state.
    """
    pipeline = create_pipeline()
    resume_state = pipeline.load_state(resume_state)
    result = pipeline.run(data={}, resume_state=resume_state)

    if "output_validator" in result and "valid_replies" in result["output_validator"]:
        valid_reply = result["output_validator"]["valid_replies"][0].text
        valid_json = json.loads(valid_reply)
        print(valid_json)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakpoint", action="store_true", help="Run pipeline with breakpoints")
    parser.add_argument("--resume", action="store_true", help="Resume pipeline from a saved state")
    parser.add_argument("--state", type=str, required=False)
    args = parser.parse_args()

    if args.breakpoint:
        breakpoint()
    elif args.resume:
        if args.state is None:
            raise ValueError("state is required when resuming, pass it with --state <state>")
        resume(args.state)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
