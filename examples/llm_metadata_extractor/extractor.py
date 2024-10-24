from haystack import Pipeline
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.converters import PyPDFToDocument
from haystack_experimental.components import LLMMetadataExtractor

PROMPT = """
You extract keywords from a document
Each keyword should be a string.
If there are no keywords, return an empty list.
If there are multiple keywords, return them as a list of strings.
The list of keywords should be returned in JSON format like in the output example.

Example:
Input: "This is a document."
Output: {"keywords": ["document"]}

Real-data:
Input: {{input_text}}
Output: 
"""

def main():

    converter = PyPDFToDocument()
    splitter = DocumentSplitter(split_by="page", split_length=1)
    cleaner = DocumentCleaner(remove_repeated_substrings=True)
    extractor = LLMMetadataExtractor(
        prompt=PROMPT,
        expected_keys=["keywords"],
        generator_api="openai",
        input_text='input_text',
        raise_on_failure=True
    )

    pipeline = Pipeline()
    pipeline.add_component(instance=converter, name="pdf_converter")
    pipeline.add_component(instance=splitter, name="splitter")
    pipeline.add_component(instance=cleaner, name="cleaner")
    pipeline.add_component(instance=extractor, name="extractor")

    pipeline.connect("pdf_converter.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "cleaner.documents")
    pipeline.connect("splitter.documents", "extractor.documents")

    # Run the pipeline
    result = pipeline.run(data={
        'pdf_converter': {'sources': ["hellofresh-se_2023.pdf"]},
        "extractor": {"start_document": 1, "end_document": 2}}
    )