import os
import pytest

from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_experimental.components import LLMMetadataExtractor


class TestLLMMetadataExtractor:

    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(prompt="prompt", expected_keys=["key1", "key2"], generator=OpenAIGenerator(), input_text="test")
        assert isinstance(extractor.builder, PromptBuilder)
        assert isinstance(extractor.generator, OpenAIGenerator)
        assert extractor.expected_keys == ["key1", "key2"]
        assert extractor.raise_on_failure is False
        assert extractor.input_text == "test"

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(prompt="prompt", expected_keys=["key1", "key2"], raise_on_failure=True, generator=OpenAIGenerator(), input_text="test")
        assert isinstance(extractor.builder, PromptBuilder)
        assert extractor.expected_keys == ["key1", "key2"]
        assert extractor.raise_on_failure is True


    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor = LLMMetadataExtractor(
            prompt="some prompt that was used with the LLM",
            expected_keys=["key1", "key2"],
            generator=OpenAIGenerator(model="gpt-4o-mini"),
            input_text="test",
            raise_on_failure=True)
        extractor_dict = extractor.to_dict()
        assert extractor_dict == {
            'type': 'haystack_experimental.components.extractors.llm_metadata_extractor.LLMMetadataExtractor',
            'init_parameters': {
                'prompt': 'some prompt that was used with the LLM',
                'expected_keys': ['key1', 'key2'],
                'raise_on_failure': True,
                'input_text': 'test',
                'generator': {
                    'type': 'haystack.components.generators.openai.OpenAIGenerator',
                    'init_parameters': {
                        'model': 'gpt-4o-mini',
                        'streaming_callback': None,
                        'api_base_url': None,
                        'organization': None,
                        'generation_kwargs': {},
                        'system_prompt': None,
                        'api_key': {'type': 'env_var', 'env_vars': ['OPENAI_API_KEY'], 'strict': True}
                    }
                }
            }
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
        extractor_dict = {
            'type': 'haystack_experimental.components.extractors.llm_metadata_extractor.LLMMetadataExtractor',
            'init_parameters': {
                'prompt': 'some prompt that was used with the LLM',
                'expected_keys': ['key1', 'key2'],
                'raise_on_failure': False,
                'input_text': 'test',
                'generator': {
                    'type': 'haystack.components.generators.openai.OpenAIGenerator',
                    'init_parameters': {
                        'model': 'gpt-4o-mini',
                        'streaming_callback': None,
                        'api_base_url': None,
                        'organization': None,
                        'generation_kwargs': {},
                        'system_prompt': None,
                        'api_key': {'type': 'env_var', 'env_vars': ['OPENAI_API_KEY'], 'strict': True}
                    }
                }
            }
        }
        extractor = LLMMetadataExtractor.from_dict(extractor_dict)
        assert extractor.raise_on_failure is False
        assert extractor.expected_keys == ["key1", "key2"]
        assert extractor.prompt == "some prompt that was used with the LLM"
        assert extractor.generator.model == "gpt-4o-mini"
        assert isinstance(extractor.generator, OpenAIGenerator)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    def test_live_run(self):
        docs = [
            Document(content="deepset was founded in 2018 in Berlin, and is known for its Haystack framework"),
            Document(content="Hugging Face is a company founded in Paris, France and is known for its Transformers library")
        ]

        ner_prompt = """
        Given a text and a list of entity types, identify all entities of those types from the text.

        -Steps-
        1. Identify all entities. For each identified entity, extract the following information:
        - entity_name: Name of the entity, capitalized
        - entity_type: One of the following types: [organization, person, product, service, industry]
        Format each entity as {"entity": <entity_name>, "entity_type": <entity_type>}

        2. Return output in a single list with all the entities identified in steps 1.

        -Examples-
        ######################
        Example 1:
        entity_types: [organization, product, service, industry, investment strategy, market trend]
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
        """

        doc_store = InMemoryDocumentStore()
        extractor = LLMMetadataExtractor(prompt=ner_prompt, expected_keys=["entities"], input_text="input_text", generator=OpenAIGenerator())
        writer = DocumentWriter(document_store=doc_store)
        pipeline = Pipeline()
        pipeline.add_component("extractor", extractor)
        pipeline.add_component("doc_writer", writer)
        pipeline.connect("extractor.documents", "doc_writer.documents")
        pipeline.run(data={"documents": docs})

        doc_store_docs = doc_store.filter_documents()
        assert len(doc_store_docs) == 2
        assert "entities" in doc_store_docs[0].meta
        assert "entities" in doc_store_docs[1].meta
