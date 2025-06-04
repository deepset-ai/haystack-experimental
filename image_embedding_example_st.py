import glob

from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_experimental.components.embedders.sentence_transformers_doc_image_embedder import (
    SentenceTransformersDocumentImageEmbedder,
)

# from sentence_transformers import SentenceTransformer, util

text_embedder = SentenceTransformersTextEmbedder(
    model="sentence-transformers/clip-ViT-B-32",
)
text_embedder.warm_up()

query = "apple"
query_embedding = text_embedder.run(query)["embedding"]

image_embedder = SentenceTransformersDocumentImageEmbedder(
    model="sentence-transformers/clip-ViT-B-32",
    meta_field_for_image_path="image_path",
)
image_embedder.warm_up()

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

# check if the results are the same of vanilla SentenceTransformer
# for doc in results["documents"]:
#     print(doc.meta)
#     print(util.cos_sim(query_embedding, doc.embedding))
