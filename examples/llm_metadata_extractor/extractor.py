from pathlib import Path

from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_experimental.components import LLMMetadataExtractor

PROMPT = """
You extract keywords from a document.
Return a list of keywords in JSON that are present in the document.
Each keyword should be a string.
If there are no keywords, return an empty list.
If there are multiple keywords, return them as a list of strings.

Example:
Input: "This is a document."
Output: {"keywords": ["document"]}

Real-data:
Input: {{input_text}}
Output: 
"""

def main():

    converter = PyPDFToDocument()
    extractor = LLMMetadataExtractor(
        prompt=PROMPT, expected_keys=["keywords"], generator_api="openai", input_text='input_text'
    )
    splitter = DocumentSplitter(split_by="page")
    doc_store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=doc_store)
    cleaner = DocumentCleaner(remove_repeated_substrings=True)

    pipeline = Pipeline()
    pipeline.add_component(instance=converter, name="pdf_converter")
    pipeline.add_component(instance=splitter, name="splitter")
    # pipeline.add_component(instance=cleaner, name="cleaner")
    # pipeline.add_component(instance=extractor, name="extractor")
    # pipeline.add_component(instance=writer, name="writer")

    pipeline.connect("pdf_converter.documents", "splitter.documents")
    # pipeline.connect("splitter.documents", "cleaner.documents")
    # pipeline.connect("splitter.documents", "extractor.documents")
    # pipeline.connect("extractor.documents", "writer.documents")

    # Run the pipeline
    pipeline.run(data={'pdf_converter': {'sources': ["file"]}})