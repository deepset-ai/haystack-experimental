import glob

from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.cohere import CohereTextEmbedder

from haystack_experimental.components.embedders.cohere_doc_image_embedder import CohereDocumentImageEmbedder

text_embedder = CohereTextEmbedder(
    model="embed-v4.0",
)


query = "apple"
query_embedding = text_embedder.run(query)["embedding"]

image_embedder = CohereDocumentImageEmbedder(
    model="embed-v4.0",
    meta_field_for_image_path="image_path",
)

docs = []
for i, image_path in enumerate(glob.glob("test/test_files/images/*")):
    docs.append(
        Document(
            content=f"Doc {i}",
            meta={"image_path": image_path},
        )
    )


docs_with_embeddings = image_embedder.run(docs)["documents"]

document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
document_store.write_documents(docs_with_embeddings)

retriever = InMemoryEmbeddingRetriever(
    document_store=document_store,
    # scale_score=True,
    return_embedding=True,
)

results = retriever.run(query_embedding)
print(results)
