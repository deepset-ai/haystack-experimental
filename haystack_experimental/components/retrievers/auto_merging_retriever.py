# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Dict, List

from haystack import DeserializationError, Document, component, default_to_dict
from haystack.core.serialization import default_from_dict, import_class_by_name
from haystack.document_stores.types import DocumentStore


@component
class AutoMergingRetriever:
    """
    A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

    The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
    are indexed in a document store. See the HierarchicalDocumentSplitter for more information on how to create
    such a structure. During retrieval, if the number of matched leaf documents below the same parent is
    higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
    documents.

    The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
    a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
    chunks alone.

    ```python

    from haystack import Document
    from haystack_experimental.components.splitters import HierarchicalDocumentBuilder
    from haystack_experimental.components.retrievers.auto_merging_retriever import AutoMergingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."

    docs = [Document(content=text)]
    builder = HierarchicalDocumentBuilder(block_sizes=[10, 3], split_overlap=0, split_by="word")
    docs = builder.run(docs)

    # store level-1 parent documents and initialize the retriever
    doc_store_parents = InMemoryDocumentStore()
    for doc in docs["documents"]:
        if doc.meta["children_ids"] and doc.meta["level"] == 1:
            doc_store_parents.write_documents([doc])
    retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

    # assume we retrieved 2 leaf docs from the same parent, the parent document should be returned,
    # since it has 3 children and the threshold=0.5, and we retrieved 2 children (2/3 > 0.66(6))
    leaf_docs = [doc for doc in docs["documents"] if not doc.meta["children_ids"]]
    docs = retriever.run(leaf_docs[4:6])
    >> {'documents': [Document(id=5384f4d58e13beb40ce80ab324a1da24f70ed69c2ec4c4f2a6f64abbc846a794,
    >> content: 'warm glow over the trees. Birds began to sing.',
    >> meta: {'block_size': 10, 'parent_id': '835b610ae31936739a47ce504674d3e86756688728b8c2b83f83484f3e1e4697',
    >> 'children_ids': ['c17e28e4b4577f892aba181a3aaa2880ef7531c8fbc5d267bda709198b3fec0b', '3ffd48a3a273ed72c83240d3f74e40cdebfb5dbc706b198d3be86ce45086593d', '3520de2d4a0c107bce7c84c181663b93b13e1a0cc0e4ea1bcafd0f9b5761ef42'],
    >> 'level': 1, 'source_id': '835b610ae31936739a47ce504674d3e86756688728b8c2b83f83484f3e1e4697',
    >> 'page_number': 1, 'split_id': 1, 'split_idx_start': 45})]}
    ```
    """  # noqa: E501

    def __init__(self, document_store: DocumentStore, threshold: float = 0.5):
        """
        Initialize the AutoMergingRetriever.

        :param document_store: DocumentStore from which to retrieve the parent documents
        :param threshold: Threshold to decide whether the parent instead of the individual documents is returned
        """

        if threshold > 1 or threshold < 0:
            raise ValueError("The threshold parameter must be between 0 and 1.")

        self.document_store = document_store
        self.threshold = threshold

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(self, document_store=docstore, threshold=self.threshold)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoMergingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        init_params = data.get("init_parameters", {})

        if "document_store" not in init_params:
            raise DeserializationError("Missing 'document_store' in serialization data")
        if "type" not in init_params["document_store"]:
            raise DeserializationError("Missing 'type' in document store's serialization data")

        # deserialize the document store
        doc_store_data = data["init_parameters"]["document_store"]
        try:
            doc_store_class = import_class_by_name(doc_store_data["type"])
        except ImportError as e:
            raise DeserializationError(f"Class '{doc_store_data['type']}' not correctly imported") from e

        if hasattr(doc_store_class, "from_dict"):
            data["init_parameters"]["document_store"] = doc_store_class.from_dict(doc_store_data)
        else:
            data["init_parameters"]["document_store"] = default_from_dict(doc_store_class, doc_store_data)

        # deserialize the component
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, matched_leaf_documents: List[Document]):
        """
        Run the AutoMergingRetriever.

        Groups the matched leaf documents by their parent documents and returns the parent documents if the number of
        matched leaf documents below the same parent is higher than the defined threshold. Otherwise, returns the
        matched leaf documents.

        :param matched_leaf_documents: List of leaf documents that were matched by a retriever
        """

        docs_to_return = []

        # group the matched leaf documents by their parent documents
        parent_documents: Dict[str, List[Document]] = defaultdict(list)
        for doc in matched_leaf_documents:
            parent_documents[doc.meta["__parent_id"]].append(doc)

        # find total number of children for each parent document
        for doc_id, retrieved_child_docs in parent_documents.items():
            parent_doc = self.document_store.filter_documents({"field": "id", "operator": "==", "value": doc_id})
            parent_children_count = len(parent_doc[0].meta["__children_ids"])

            # return either the parent document or the matched leaf documents based on the threshold value
            score = len(retrieved_child_docs) / parent_children_count
            if score >= self.threshold:
                # return the parent document
                docs_to_return.append(parent_doc[0])
            else:
                # return all the matched leaf documents which are child of this parent document
                leafs_ids = [doc.id for doc in retrieved_child_docs]
                docs_to_return.extend([doc for doc in matched_leaf_documents if doc.id in leafs_ids])

        return {"documents": docs_to_return}
