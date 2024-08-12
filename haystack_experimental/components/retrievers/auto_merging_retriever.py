from collections import defaultdict
from typing import Dict, List

from haystack import Document, component
from haystack.document_stores.types import DocumentStore


@component
class AutoMergingRetriever:
    """
    A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

    The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
    are indexed in a document store. During retrieval, if the number of matched leaf documents below the same parent is
    higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
    documents.

    The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
    a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
    chunks alone.
    """

    def __init__(self, document_store: DocumentStore, threshold: float = 0.5):
        """
        Initialize the AutoMergingRetriever.

        Groups the matched leaf documents by their parent documents and returns the parent documents if the number of
        matched leaf documents below the same parent is higher than the defined threshold. Otherwise, returns the
        matched leaf documents.

        :param document_store: DocumentStore from which to retrieve the parent documents
        :param threshold: Threshold to decide whether the parent instead of the individual documents is returned
        """

        if threshold > 1 or threshold < 0:
            raise ValueError("The threshold parameter must be between 0 and 1.")

        self.document_store = document_store
        self.threshold = threshold

    @component.output_types(documents=List[Document])
    def run(self, matched_leaf_documents: List[Document]):
        """
        Run the AutoMergingRetriever.

        :param matched_leaf_documents: List of leaf documents that were matched by a retriever
        """

        docs_to_return = []

        # group the matched leaf documents by their parent documents
        parent_documents: Dict[str, List[Document]] = defaultdict(list)
        for doc in matched_leaf_documents:
            parent_documents[doc.meta["parent_id"]].append(doc)

        print(parent_documents)

        # find total number of children for each parent document
        for doc_id in parent_documents.keys():
            parent_doc = self.document_store.filter_documents({"field": "id", "operator": "==", "value": doc_id})
            parent_children_count = len(parent_doc[0].meta["children_ids"])

            # return either the parent document or the matched leaf documents based on the threshold value
            print(parent_children_count, len(parent_documents[doc_id]), self.threshold)
            if len(parent_documents[doc_id]) / parent_children_count >= self.threshold:
                # return the parent document
                docs_to_return.append(parent_doc[0])
            else:
                # retrieve all the matched leaf documents for this parent
                leafs_ids = [doc.id for doc in parent_documents[doc_id]]
                docs_to_return.extend([doc for doc in matched_leaf_documents if doc.id in leafs_ids])

        return {"documents": docs_to_return}
