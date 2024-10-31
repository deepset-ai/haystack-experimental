# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.builders import PromptBuilder
from haystack.components.generators import AzureOpenAIGenerator, OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_secrets_inplace

from haystack_experimental.util.utils import expand_page_range

with LazyImport(message="Run 'pip install \"amazon-bedrock-haystack==1.0.2\"'") as amazon_bedrock_generator:
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockGenerator

with LazyImport(message="Run 'pip install \"google-vertex-haystack==2.0.0\"'") as vertex_ai_gemini_generator:
    from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator


logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """
    Currently LLM providers supported by `LLMMetadataExtractor`.
    """

    OPENAI = "openai"
    OPENAI_AZURE = "openai_azure"
    AWS_BEDROCK = "aws_bedrock"
    GOOGLE_VERTEX = "google_vertex"

    @staticmethod
    def from_str(string: str) -> "LLMProvider":
        """
        Convert a string to a LLMProvider enum.
        """
        provider_map = {e.value: e for e in LLMProvider}
        provider = provider_map.get(string)
        if provider is None:
            msg = (
                f"Invalid LLMProvider '{string}'"
                f"Supported LLMProviders are: {list(provider_map.keys())}"
            )
            raise ValueError(msg)
        return provider


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

    extractor = LLMMetadataExtractor(prompt=NER_PROMPT, expected_keys=["entities"], generator_api="openai", prompt_variable='input_text')
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

    def __init__( # pylint: disable=R0917
        self,
        prompt: str,
        prompt_variable: str,
        expected_keys: List[str],
        generator_api: Union[str,LLMProvider],
        generator_api_params: Optional[Dict[str, Any]] = None,
        page_range: Optional[List[Union[str, int]]] = None,
        raise_on_failure: bool = False,
    ):
        """
        Initializes the LLMMetadataExtractor.

        :param prompt: The prompt to be used for the LLM.
        :param prompt_variable: The variable in the prompt to be processed by the PromptBuilder.
        :param expected_keys: The keys expected in the JSON output from the LLM.
        :param generator_api: The API provider for the LLM. Currently supported providers are:
                              "openai", "openai_azure", "aws_bedrock", "google_vertex"
        :param generator_api_params: The parameters for the LLM generator.
        :param page_range: A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
                           metadata from the first and third pages of each document. It also accepts printable range
                           strings, e.g.: ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,
                           11, 12. If None, metadata will be extracted from the entire document for each document in the
                           documents list.
                           This parameter is optional and can be overridden in the `run` method.
        :param raise_on_failure: Whether to raise an error on failure to validate JSON output.
        :returns:

        """
        self.prompt = prompt
        self.prompt_variable = prompt_variable
        self.builder = PromptBuilder(prompt, required_variables=[prompt_variable])
        self.raise_on_failure = raise_on_failure
        self.expected_keys = expected_keys
        self.generator_api = generator_api if isinstance(generator_api, LLMProvider)\
            else LLMProvider.from_str(generator_api)
        self.generator_api_params = generator_api_params or {}
        self.llm_provider = self._init_generator(self.generator_api, self.generator_api_params)
        if self.prompt_variable not in self.prompt:
            raise ValueError(f"Prompt variable '{self.prompt_variable}' must be in the prompt.")
        self.splitter = DocumentSplitter(split_by="page", split_length=1)
        self.expanded_range = expand_page_range(page_range) if page_range else None

    @staticmethod
    def _init_generator(
            generator_api: LLMProvider,
            generator_api_params: Optional[Dict[str, Any]]
    ) -> Union[OpenAIGenerator, AzureOpenAIGenerator, "AmazonBedrockGenerator", "VertexAIGeminiGenerator"]:
        """
        Initialize the chat generator based on the specified API provider and parameters.
        """
        if generator_api == LLMProvider.OPENAI:
            return OpenAIGenerator(**generator_api_params)
        if generator_api == LLMProvider.OPENAI_AZURE:
            return AzureOpenAIGenerator(**generator_api_params)
        if generator_api == LLMProvider.AWS_BEDROCK:
            amazon_bedrock_generator.check()
            return AmazonBedrockGenerator(**generator_api_params)
        if generator_api == LLMProvider.GOOGLE_VERTEX:
            vertex_ai_gemini_generator.check()
            return VertexAIGeminiGenerator(**generator_api_params)
        raise ValueError(f"Unsupported generator API: {generator_api}")

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
            msg = "Response from LLM is not a valid JSON."
            if self.raise_on_failure:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        if not all(output in parsed_output for output in expected):
            msg = f"Expected response from LLM to be a JSON with keys {expected}, got {received}."
            if self.raise_on_failure:
                raise ValueError(msg)
            logger.warning(msg)
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        llm_provider = self.llm_provider.to_dict()

        return default_to_dict(
            self,
            prompt=self.prompt,
            input_text=self.prompt_variable,
            expected_keys=self.expected_keys,
            raise_on_failure=self.raise_on_failure,
            generator_api=self.generator_api.value,
            generator_api_params=llm_provider["init_parameters"],
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

        init_parameters = data.get("init_parameters", {})
        if "generator_api" in init_parameters:
            data["init_parameters"]["generator_api"] = LLMProvider.from_str(data["init_parameters"]["generator_api"])
        if "generator_api_params" in init_parameters:
            deserialize_secrets_inplace(data["init_parameters"]["generator_api_params"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _extract_metadata_and_update_doc(self, document: Document, content: str):
        """
        Extract metadata from the content and updates the document's metadata with the extracted metadata.

        If the extraction fails, i.e.: no JSON is returned by the LLM API, the error message will be stored in
        `errors`.

        :param document: Document to be updated with the extracted metadata.
        :param content: Content to extract metadata from.
        """
        prompt_with_doc = self.builder.run(
            template=self.prompt,
            template_variables={self.prompt_variable:content}
        )
        result = self.llm_provider.run(prompt=prompt_with_doc["prompt"])
        llm_answer = result["replies"][0]
        if self.is_valid_json_and_has_expected_keys(expected=self.expected_keys, received=llm_answer):
            extracted_metadata = json.loads(llm_answer)
            for k in self.expected_keys:
                document.meta[k] = extracted_metadata[k]

    @component.output_types(documents=List[Document], errors=Dict[str, Any])
    def run(self, documents: List[Document], page_range: Optional[List[Union[str, int]]] = None):
        """
        Extract metadata from documents using a Language Model.

        If `page_range` is provided, the metadata will be extracted from the specified range of pages. This component
        will split the documents into pages and extract metadata from the specified range of pages. The metadata will be
        extracted from the entire document if `page_range` is not provided.

        The original documents will be returned  updated with the extracted metadata.

        :param documents: List of documents to extract metadata from.
        :param page_range: A range of pages to extract metadata from. For example, page_range=['1', '3'] will extract
                           metadata from the first and third pages of each document. It also accepts printable range
                           strings, e.g.: ['1-3', '5', '8', '10-12'] will extract metadata from pages 1, 2, 3, 5, 8, 10,
                           11, 12.
                           If None, metadata will be extracted from the entire document for each document in the
                           documents list.
        :returns:
            A dictionary with the keys:
            - "documents": The original list of documents updated with the extracted metadata.
            - "errors": A dictionary with document IDs as keys and error messages as values.
        """

        errors: Dict[str, Any] = {}

        for document in documents:

            if not document.content:
                logger.warning(f"Document {document.id} has no content. Skipping metadata extraction.")
                continue

            if self.expanded_range or page_range: # extract metadata from a specific range of pages

                expanded_range = self.expanded_range
                if page_range:
                    expanded_range = expand_page_range(page_range)

                if not self.expanded_range:
                    msg = f"Page range {self.expanded_range} invalid"
                    raise ValueError(msg)

                pages = self.splitter.run(documents=[document])
                content = [p.content + "\f" for idx, p in enumerate(pages["documents"]) if (idx + 1) in self.expanded_range]

                self._extract_metadata_and_update_doc(document, errors, "".join(content))

            else:
                # extract metadata from the entire document
                self._extract_metadata_and_update_doc(document, errors, document.content)

        return {"documents": documents, "errors": errors}
