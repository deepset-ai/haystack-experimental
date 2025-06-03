# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document, Pipeline
from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers.document_writer import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_experimental.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator
from haystack_experimental.components.image_converters.document_to_image import DocumentToImageContent

# Initialize the document store and document writer
document_store = InMemoryDocumentStore()
document_writer = DocumentWriter(document_store=document_store)

# Convert a PDF and JPG into Documents
pdf_splitters = DocumentSplitter(split_by="page", split_length=1)
pdf_doc = PyPDFToDocument(store_full_path=True).run(sources=["test/test_files/pdf/sample_pdf_1.pdf"])["documents"][0]
pdf_docs = pdf_splitters.run(documents=[pdf_doc])["documents"]
image_doc = Document(
    content="This is a picture of a red apple.", meta={"file_path": "test/test_files/images/apple.jpg"}
)

# Write the documents to the document store
docs = pdf_docs + [image_doc]
document_writer.run(documents=docs)

# Create the Retrieval + Query pipeline
retriever = InMemoryBM25Retriever(document_store=document_store, top_k=2)
doc_to_image = DocumentToImageContent(detail="auto")
chat_prompt_builder = ChatPromptBuilder(
    required_variables="*",
#     template="""{% message role="system" %}
# You are a friendly assistant that answers questions based on provided documents.
# {% endmessage %}
#
# {%- message role="user" -%}
# Only provide an answer to the question using the images and text passages provided.
#
# These are the text version of the documents:
# {%- for doc in documents %}
# Document [{{ loop.index }}] :
# Relates to image: [{{ loop.index }}]
# {{ doc.content }}
# {% endfor -%}
#
# Question: {{ question }}
# Answer:
#
# {%- for img in image_contents -%}
#   {{ img | templatize_part }}
# {%- endfor -%}
# {%- endmessage -%}
# """
    template="""{% message role="system" %}
You are a friendly assistant that answers questions based on user provided images.
{% endmessage %}

{%- message role="user" -%}
Only provide an answer to the question using the images provided.

Question: {{ question }}
Answer:

{%- for img in image_contents -%}
  {{ img | templatize_part }}
{%- endfor -%}
{%- endmessage -%}
"""
)
llm = OpenAIChatGenerator()

# Create the pipeline
pipe = Pipeline()
pipe.add_component("retriever", retriever)
pipe.add_component("doc_to_image", doc_to_image)
pipe.add_component("chat_prompt_builder", chat_prompt_builder)
pipe.add_component("llm", llm)

pipe.connect("retriever.documents", "doc_to_image.documents")
# pipe.connect("doc_to_image.image_documents", "chat_prompt_builder.documents")
pipe.connect("doc_to_image.image_contents", "chat_prompt_builder.image_contents")
pipe.connect("chat_prompt_builder.prompt", "llm.messages")

# Run the pipeline with a query
query = "What is the color of the apple in the image?"
result = pipe.run(
    data={"retriever": {"query": query}, "chat_prompt_builder": {"question": query}},
    include_outputs_from={"chat_prompt_builder"},
)
