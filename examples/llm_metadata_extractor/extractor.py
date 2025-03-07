from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument

from haystack_experimental.components import LLMMetadataExtractor

PROMPT = """
You extract keywords from a document
Each keyword should be a string.
If there are no keywords, return an empty list.
If there are multiple keywords, return them as a list of strings.
The list of keywords should be returned in JSON format like in the output example.
Don't include the word 'json' in the output. Don't include any other information in the output.

Example:
Input: "This is a document."
Output: {"keywords": ["document"]}

Real-data:
Input: {{input_text}}
Output: 
"""

converter = PyPDFToDocument()
extractor = LLMMetadataExtractor(
    prompt=PROMPT,
    expected_keys=["keywords"],
    generator_api="openai",
    prompt_variable="input_text",
    raise_on_failure=True
)

pipeline = Pipeline()
pipeline.add_component(instance=converter, name="pdf_converter")
pipeline.add_component(instance=extractor, name="extractor")

pipeline.connect("pdf_converter.documents", "extractor.documents")

# Run the pipeline
result = pipeline.run(data={
    "pdf_converter": {"sources": ["hellofresh-se_2023.pdf", "NYSE_RHT_2019.pdf"]},
    "extractor": {"page_range": [1,10,20,30,40,50]}}
)
