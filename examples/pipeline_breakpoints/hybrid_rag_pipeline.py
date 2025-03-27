import argparse
from haystack_experimental.core.pipeline.pipeline import Pipeline

from haystack import Document
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy


def indexing():
    """
    Indexing documents in a DocumentStore.
    """

    print("Indexing documents...")

    documents = [
        Document(content="My name is Jean and I live in Paris."),
        Document(content="My name is Mark and I live in Berlin."),
        Document(content="My name is Giorgio and I live in Rome."),
    ]

    document_store = InMemoryDocumentStore()
    doc_writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2", progress_bar=False)

    ingestion_pipe = Pipeline()
    ingestion_pipe.add_component(instance=doc_embedder, name="doc_embedder")
    ingestion_pipe.add_component(instance=doc_writer, name="doc_writer")

    ingestion_pipe.connect("doc_embedder.documents", "doc_writer.documents")
    ingestion_pipe.run({"doc_embedder": {"documents": documents}})

    return document_store


def hybrid_retrieval(doc_store):
    """
    A simple pipeline for hybrid retrieval using BM25 and embeddings.
    """

    query_embedder = SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", progress_bar=False)

    # Build a RAG pipeline with a Retriever to get relevant documents to the query and a OpenAIGenerator interacting
    # with LLMs using a custom prompt.
    prompt_template = """
    Given these documents, answer the question based on the document content only.\nDocuments:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    \nQuestion: {{question}}
    \nAnswer:
    """
    rag_pipeline = Pipeline()
    rag_pipeline.add_component(instance=InMemoryBM25Retriever(document_store=doc_store), name="bm25_retriever")
    rag_pipeline.add_component(instance=query_embedder, name="query_embedder")
    rag_pipeline.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=doc_store),
        name="embedding_retriever"
    )
    rag_pipeline.add_component(instance=DocumentJoiner(sort_by_score=False), name="doc_joiner")
    rag_pipeline.add_component(
        instance=TransformersSimilarityRanker(model="intfloat/simlm-msmarco-reranker", top_k=5),
        name="ranker"
    )
    rag_pipeline.add_component(instance=PromptBuilder(
        template=prompt_template, required_variables=['documents', 'question']),
        name="prompt_builder"
    )
    rag_pipeline.add_component(instance=OpenAIGenerator(), name="llm")
    rag_pipeline.add_component(instance=AnswerBuilder(), name="answer_builder")

    rag_pipeline.connect("query_embedder", "embedding_retriever.query_embedding")
    rag_pipeline.connect("embedding_retriever", "doc_joiner.documents")
    rag_pipeline.connect("bm25_retriever", "doc_joiner.documents")
    rag_pipeline.connect("doc_joiner", "ranker.documents")
    rag_pipeline.connect("ranker", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    rag_pipeline.connect("llm.replies", "answer_builder.replies")
    rag_pipeline.connect("llm.meta", "answer_builder.meta")
    rag_pipeline.connect("doc_joiner", "answer_builder.documents")

    return rag_pipeline


def breakpoint():
    doc_store = indexing()
    pipeline = hybrid_retrieval(doc_store)

    question = "Where does Mark live?"
    data = {
        "query_embedder": {"text": question},
        "bm25_retriever": {"query": question},
        "ranker": {"query": question, "top_k": 10},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
    }
    print("\n\nStarting pipeline...")
    pipeline.run(data, breakpoints={("query_embedder", 0)})


def resume(resume_state):
    doc_store = indexing()
    pipeline = hybrid_retrieval(doc_store)
    print("\n\nResuming pipeline...")
    resume_state = pipeline.load_state(resume_state)
    result = pipeline.run(data={}, resume_state=resume_state)
    print(result['answer_builder']['answers'][0].data)
    print(result['answer_builder']['answers'][0].meta)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breakpoint", action="store_true", help="Run pipeline with breakpoints")
    parser.add_argument("--resume", action="store_true", help="Resume pipeline from a saved state")
    parser.add_argument("--state", type=str, required=False)
    args = parser.parse_args()

    if args.breakpoint:
        breakpoint()

    elif args.resume:
        if args.state is None:
            raise ValueError("state is required when resuming, pass it with --state <state>")
        resume(args.state)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()