# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from haystack import Document, component, logging
from haystack.components.embedders.types.protocol import TextEmbedder
from haystack.components.preprocessors.sentence_tokenizer import Language, SentenceSplitter, nltk_imports
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.utils.deserialization import deserialize_component_inplace
from more_itertools import windowed

logger = logging.getLogger(__name__)


@component
class EmbeddingBasedDocumentSplitter:
    """
    Splits documents based on embedding similarity using cosine distances between sequential sentence groups.

    This component first splits text into sentences, optionally groups them, calculates embeddings for each group,
    and then uses cosine distance between sequential embeddings to determine split points. Any distance above
    the specified percentile is treated as a break point.

    This component is inspired by [5 Levels of Text Splitting](
        https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb
    ) by Greg Kamradt.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.preprocessors import EmbeddingBasedDocumentSplitter
    from haystack.components.embedders import SentenceTransformersTextEmbedder

    doc = Document(
        content="This is a first sentence. This is a second sentence. This is a third sentence. "
        "Completely different topic. The same completely different topic."
    )

    embedder = SentenceTransformersTextEmbedder()

    splitter = EmbeddingBasedDocumentSplitter(
        text_embedder=embedder,
        sentences_per_group=2,
        percentile=0.95,
        min_length=50,
        max_length=1000
    )
    splitter.warm_up()
    result = splitter.run(documents=[doc])
    ```
    """

    def __init__(
        self,
        text_embedder: TextEmbedder,
        sentences_per_group: int = 1,
        percentile: float = 0.95,
        min_length: int = 50,
        max_length: int = 1000,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
    ):
        """
        Initialize EmbeddingBasedDocumentSplitter.

        :param text_embedder: The TextEmbedder to use for calculating embeddings.
        :param sentences_per_group: Number of sentences to group together before embedding. Default is 1.
        :param percentile: Percentile threshold for cosine distance. Distances above this percentile
            are treated as break points. Default is 0.95.
        :param min_length: Minimum length of splits in characters. Splits below this length will be merged.
            Default is 50.
        :param max_length: Maximum length of splits in characters. Splits above this length will be recursively split.
            Default is 1000.
        :param language: Language for sentence tokenization. Default is "en".
        :param use_split_rules: Whether to use additional split rules for sentence tokenization. Default is True.
        :param extend_abbreviations: Whether to extend NLTK abbreviations. Default is True.
        """
        self.text_embedder = text_embedder
        self.sentences_per_group = sentences_per_group
        self.percentile = percentile
        self.min_length = min_length
        self.max_length = max_length
        self.language = language
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations

        self._init_validation()
        self.sentence_splitter: Optional[SentenceSplitter] = None

    def _init_validation(self) -> None:
        """
        Validates initialization parameters.
        """
        if self.sentences_per_group <= 0:
            raise ValueError("sentences_per_group must be greater than 0.")

        if not 0.0 <= self.percentile <= 1.0:
            raise ValueError("percentile must be between 0.0 and 1.0.")

        if self.min_length < 0:
            raise ValueError("min_length must be greater than or equal to 0.")

        if self.max_length <= self.min_length:
            raise ValueError("max_length must be greater than min_length.")

    def warm_up(self):
        """
        Warm up the component by initializing the sentence splitter.
        """
        nltk_imports.check()
        self.sentence_splitter = SentenceSplitter(
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
            keep_white_spaces=True,
        )
        self.text_embedder.warm_up()

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Split documents based on embedding similarity.

        :param documents: The documents to split.
        :returns: A dictionary with the following key:
            - `documents`: List of documents with the split texts. Each document includes:
                - A metadata field `source_id` to track the original document.
                - A metadata field `split_id` to track the split number.
                - All other metadata copied from the original document.

        :raises:
            - `RuntimeError`: If the component wasn't warmed up.
            - `TypeError`: If the input is not a list of Documents.
            - `ValueError`: If the document content is None or empty.
        """
        if self.sentence_splitter is None:
            raise RuntimeError(
                "The component EmbeddingBasedDocumentSplitter wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError("EmbeddingBasedDocumentSplitter expects a List of Documents as input.")

        split_docs: List[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"EmbeddingBasedDocumentSplitter only works with text documents but content for "
                    f"document ID {doc.id} is None."
                )
            if doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue

            doc_splits = self._split_document(doc)
            split_docs.extend(doc_splits)

        return {"documents": split_docs}

    def _split_document(self, doc: Document) -> List[Document]:
        """
        Split a single document based on embedding similarity.
        """
        sentences_result = self.sentence_splitter.split_sentences(doc.content)
        sentences = [sentence["sentence"] for sentence in sentences_result]

        sentence_groups = self._group_sentences(sentences)

        embeddings = self._calculate_embeddings(sentence_groups)

        split_points = self._find_split_points(embeddings)

        splits = self._create_splits_from_points(sentence_groups, split_points)

        # Merge small splits and split large splits
        final_splits = self._post_process_splits(splits)

        return self._create_documents_from_splits(final_splits, doc)

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """
        Group sentences into groups of sentences_per_group.
        """
        if self.sentences_per_group == 1:
            return sentences

        groups = []
        for i in range(0, len(sentences), self.sentences_per_group):
            group = sentences[i : i + self.sentences_per_group]
            groups.append(" ".join(group))

        return groups

    def _calculate_embeddings(self, sentence_groups: List[str]) -> List[List[float]]:
        """
        Calculate embeddings for each sentence group.
        """
        embeddings = []
        for group in sentence_groups:
            result = self.text_embedder.run(group)
            embeddings.append(result["embedding"])
        return embeddings

    def _find_split_points(self, embeddings: List[List[float]]) -> List[int]:
        """
        Find split points based on cosine distances between sequential embeddings.
        """
        if len(embeddings) <= 1:
            return []

        # Calculate cosine distances between sequential pairs
        distances = []
        for i in range(len(embeddings) - 1):
            distance = self._cosine_distance(embeddings[i], embeddings[i + 1])
            distances.append(distance)

        # Calculate threshold based on percentile
        threshold = np.percentile(distances, self.percentile * 100)

        # Find indices where distance exceeds threshold
        split_points = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                split_points.append(i + 1)  # +1 because we want to split after this point

        return split_points

    def _cosine_distance(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine distance between two embeddings.
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 1.0

        cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)

        return 1.0 - cosine_sim

    def _create_splits_from_points(self, sentence_groups: List[str], split_points: List[int]) -> List[str]:
        """
        Create splits based on split points.
        """
        if not split_points:
            return [" ".join(sentence_groups)]

        splits = []
        start = 0

        for point in split_points:
            split_text = " ".join(sentence_groups[start:point])
            if split_text.strip():
                splits.append(split_text)
            start = point

        # Add the last split
        if start < len(sentence_groups):
            split_text = " ".join(sentence_groups[start:])
            if split_text.strip():
                splits.append(split_text)

        return splits

    def _post_process_splits(self, splits: List[str]) -> List[str]:
        """
        Post-process splits: merge small ones and split large ones.
        """
        if not splits:
            return splits

        merged_splits = self._merge_small_splits(splits)

        final_splits = self._split_large_splits(merged_splits)

        return final_splits

    def _merge_small_splits(self, splits: List[str]) -> List[str]:
        """
        Merge splits that are below min_length.
        """
        if not splits:
            return splits

        merged = []
        current_split = splits[0]

        for split in splits[1:]:
            if len(current_split) < self.min_length:
                # Merge with next split
                current_split += " " + split
            else:
                # Current split is long enough, save it and start a new one
                merged.append(current_split)
                current_split = split

        # Don't forget the last split
        merged.append(current_split)

        return merged

    def _split_large_splits(self, splits: List[str]) -> List[str]:
        """
        Recursively split splits that are above max_length.
        """
        final_splits = []

        for split in splits:
            if len(split) <= self.max_length:
                final_splits.append(split)
            else:
                # Recursively split large splits
                # For simplicity, split by sentences first
                sentences_result = self.sentence_splitter.split_sentences(split)
                sentences = [sentence["sentence"] for sentence in sentences_result]

                # Group sentences and repeat the embedding-based splitting
                sentence_groups = self._group_sentences(sentences)
                embeddings = self._calculate_embeddings(sentence_groups)
                split_points = self._find_split_points(embeddings)
                sub_splits = self._create_splits_from_points(sentence_groups, split_points)

                # Recursively process sub-splits
                final_splits.extend(self._post_process_splits(sub_splits))

        return final_splits

    def _create_documents_from_splits(self, splits: List[str], original_doc: Document) -> List[Document]:
        """
        Create Document objects from splits.
        """
        documents = []
        metadata = deepcopy(original_doc.meta)
        metadata["source_id"] = original_doc.id

        for i, split_text in enumerate(splits):
            split_meta = deepcopy(metadata)
            split_meta["split_id"] = i
            doc = Document(content=split_text, meta=split_meta)
            documents.append(doc)

        return documents

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.
        """
        return default_to_dict(
            self,
            text_embedder=component_to_dict(obj=self.text_embedder, name="text_embedder"),
            sentences_per_group=self.sentences_per_group,
            percentile=self.percentile,
            min_length=self.min_length,
            max_length=self.max_length,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingBasedDocumentSplitter":
        """
        Deserializes the component from a dictionary.
        """
        deserialize_component_inplace(data["init_parameters"], key="text_embedder")
        return default_from_dict(cls, data)
