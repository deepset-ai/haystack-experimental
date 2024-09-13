# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import json

from typing import Optional, List, Dict, Any
from warnings import warn

from haystack import Document, component, default_to_dict, default_from_dict
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator


@component
class LLMMetadataExtractor:

    def __init__(self, prompt: str, expected_keys: List[str], model: Optional[str] = None, raise_on_failure: bool = True):
        self.prompt = prompt
        self.builder = PromptBuilder(prompt)
        self.generator = OpenAIGenerator() if model is None else OpenAIGenerator(model=model)
        self.expected_keys = expected_keys
        self.raise_on_failure = raise_on_failure

    def is_valid_json_and_has_expected_keys(self, expected: List[str], received: str) -> bool:
        """
        Output must be a valid JSON with the expected keys.

        :param expected:
            Names of expected outputs
        :param received:
            Names of received outputs

        :raises ValueError:
            If the output is not a valid JSON with the expected keys:
            - with `raise_on_failure` set to True a ValueError is raised.
            - with `raise_on_failure` set to False a warning is issued and False is returned.

        :returns:
            True if the received output is a valid JSON with the expected keys, False otherwise.
        """
        try:
            parsed_output = json.loads(received)
        except json.JSONDecodeError:
            msg = "Response from LLM evaluator is not a valid JSON."
            if self.raise_on_failure:
                raise ValueError(msg)
            warn(msg)
            return False

        if not all(output in parsed_output for output in expected):
            msg = f"Expected response from LLM evaluator to be JSON with keys {expected}, got {received}."
            if self.raise_on_failure:
                raise ValueError(msg)
            warn(msg)
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, prompt=self.prompt, expected_keys=self.expected_keys, raise_on_failure=self.raise_on_failure, model=self.generator.model)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMetadataExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        return default_from_dict(cls, data)

    @component.output_types(documents_meta=List[Document])
    def run(self, documents: List[Document]):
        for document in documents:
            prompt_with_doc = self.builder.run(input_text=document.content)
            result = self.generator.run(prompt=prompt_with_doc['prompt'])
            llm_answer = result["replies"][0]
            if self.is_valid_json_and_has_expected_keys(expected=self.expected_keys, received=llm_answer):
                extracted_metadata = json.loads(llm_answer)
                for k in self.expected_keys:
                    document.meta[k] = extracted_metadata[k]
        return {"documents_meta": documents}
