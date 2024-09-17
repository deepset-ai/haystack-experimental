# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import json
import logging
from typing import Any, Dict, List, Tuple, Union, Type
from warnings import warn

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.builders import PromptBuilder
from haystack.components.generators import AzureOpenAIGenerator, OpenAIGenerator
from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator

from haystack.utils import deserialize_document_store_in_init_params_inplace

logger = logging.getLogger(__name__)

SUPPORTED_GENERATORS = (OpenAIGenerator, AzureOpenAIGenerator, AmazonBedrockGenerator, VertexAIGeminiGenerator)

SUPPORTED_GENERATORS_TYPES: List[
    Type[Union[OpenAIGenerator, AzureOpenAIGenerator, AmazonBedrockGenerator, VertexAIGeminiGenerator]]] = \
    [OpenAIGenerator, AzureOpenAIGenerator, AmazonBedrockGenerator, VertexAIGeminiGenerator]


@component
class LLMMetadataExtractor:
    """
    Extracts metadata from documents using a Large Language Model (LLM) from OpenAI.

    The metadata is extracted by providing a prompt to n LLM that generates the metadata.

    ```python
    from haystack import Document
    from haystack.components.generators import OpenAIGenerator
    from haystack_experimental.components.extractors import LLMMetadataExtractor

    NER_PROMPT = '''
    -Goal-
    Given text and a list of entity types, identify all entities of those types from the text.

    -Steps-
    1. Identify all entities. For each identified entity, extract the following information:
    - entity_name: Name of the entity, capitalized
    - entity_type: One of the following types: [organization, product, service, industry]
    Format each entity as {"entity": <entity_name>, "entity_type": <entity_type>}

    2. Return output in a single list with all the entities identified in steps 1.

    -Examples-
    ######################
    Example 1:
    entity_types: [organization, person, partnership, financial metric, product, service, industry, investment strategy, market trend]
    text:
    Another area of strength is our co-brand issuance. Visa is the primary network partner for eight of the top 10 co-brand partnerships in the US today and we are pleased that Visa has finalized a multi-year extension of our successful credit co-branded partnership with Alaska Airlines, a portfolio that benefits from a loyal customer base and high cross-border usage.
    We have also had significant co-brand momentum in CEMEA. First, we launched a new co-brand card in partnership with Qatar Airways, British Airways and the National Bank of Kuwait. Second, we expanded our strong global Marriott relationship to launch Qatar's first hospitality co-branded card with Qatar Islamic Bank. Across the United Arab Emirates, we now have exclusive agreements with all the leading airlines marked by a recent agreement with Emirates Skywards.
    And we also signed an inaugural Airline co-brand agreement in Morocco with Royal Air Maroc. Now newer digital issuers are equally
    ------------------------
    output:
    {"entities": [{"entity": "Visa", "entity_type": "company"}, {"entity": "Alaska Airlines", "entity_type": "company"}, {"entity": "Qatar Airways", "entity_type": "company"}, {"entity": "British Airways", "entity_type": "company"}, {"entity": "National Bank of Kuwait", "entity_type": "company"}, {"entity": "Marriott", "entity_type": "company"}, {"entity": "Qatar Islamic Bank", "entity_type": "company"}, {"entity": "Emirates Skywards", "entity_type": "company"}, {"entity": "Royal Air Maroc", "entity_type": "company"}]}
    #############################
    -Real Data-
    ######################
    entity_types: [company, organization, person, country, product, service]
    text: {{input_text}}
    ######################
    output:
    '''

    docs = [
        Document(content="deepset was founded in 2018 in Berlin, and is known for its Haystack framework"),
        Document(content="Hugging Face is a company founded in Paris, France and is known for its Transformers library")
    ]

    extractor = LLMMetadataExtractor(prompt=NER_PROMPT, expected_keys=["entities"], generator=OpenAIGenerator(), input_text='input_text')
    extractor.run(documents=docs)
    >> {'documents': [
        Document(id=.., content: 'deepset was founded in 2018 in Berlin, and is known for its Haystack framework',
        meta: {'entities': [{'entity': 'deepset', 'entity_type': 'company'}, {'entity': 'Berlin', 'entity_type': 'city'},
              {'entity': 'Haystack', 'entity_type': 'product'}]}),
        Document(id=.., content: 'Hugging Face is a company founded in Paris, France and is known for its Transformers library',
        meta: {'entities': [
                {'entity': 'Hugging Face', 'entity_type': 'company'}, {'entity': 'Paris', 'entity_type': 'city'},
                {'entity': 'France', 'entity_type': 'country'}, {'entity': 'Transformers', 'entity_type': 'product'}
                ]})
           ]
       }
    >>
    ```
    """ # noqa: E501
    def __init__(
        self,
        prompt: str,
        input_text: str,
        expected_keys: List[str],
        generator: SUPPORTED_GENERATORS_TYPES,
        raise_on_failure: bool = False,
    ):
        """
        Initializes the LLMMetadataExtractor.

        :param prompt: The prompt to be used for the LLM.
        :param input_text: The input text to be processed by the PromptBuilder.
        :param expected_keys: The keys expected in the JSON output from the LLM.
        :param generator: The generator to be used for generating responses from the LLM. Currently, supports OpenAI,
                          Azure OpenAI, and Amazon Bedrock.
        :param raise_on_failure: Whether to raise an error on failure to validate JSON output.
        :returns:

        """
        self.prompt = prompt
        self.input_text = input_text
        self.builder = PromptBuilder(prompt, required_variables=[input_text])
        self.raise_on_failure = raise_on_failure
        self.expected_keys = expected_keys
        self.generator = generator
        self._check_llm()

    def _check_prompt(self):
        if self.input_text not in self.prompt:
            raise ValueError(f"{self.input_text} must be in the prompt.")

    def _check_llm(self):
        if not isinstance(self.generator, SUPPORTED_GENERATORS):
            raise ValueError(
                "Generator must be an instance of OpenAIGenerator, AzureOpenAIGenerator, "
                "AmazonBedrockGenerator or VertexAIGeminiGenerator."
            )

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
            logger.warning(msg)
            return False

        if not all(output in parsed_output for output in expected):
            msg = f"Expected response from LLM evaluator to be JSON with keys {expected}, got {received}."
            if self.raise_on_failure:
                raise ValueError(msg)
            warn(msg)
            logger.warning(msg)
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            prompt=self.prompt,
            input_text=self.input_text,
            expected_keys=self.expected_keys,
            raise_on_failure=self.raise_on_failure,
            generator=self.generator.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMetadataExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        deserialize_document_store_in_init_params_inplace(data, key="generator")
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document], errors=List[Tuple[str,Any]])
    def run(self, documents: List[Document]) -> Dict[str, Union[List[Document], List[Tuple[str, Any]]]]:
        """
        Extract metadata from documents using a Language Model.

        :param documents: List of documents to extract metadata from.
        :returns:
            A dictionary with the key "documents_meta" containing the documents with extracted metadata.
        """
        errors = []
        for document in documents:
            prompt_with_doc = self.builder.run(input_text=document.content)
            result = self.generator.run(prompt=prompt_with_doc["prompt"])
            llm_answer = result["replies"][0]
            if self.is_valid_json_and_has_expected_keys(expected=self.expected_keys, received=llm_answer):
                extracted_metadata = json.loads(llm_answer)
                for k in self.expected_keys:
                    document.meta[k] = extracted_metadata[k]
                errors.append((document.id, None))
            else:
                errors.append((document.id, llm_answer))

        return {"documents": documents, "errors": errors}
