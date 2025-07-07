# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch

from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_experimental.components.preprocessors import EmbeddingBasedDocumentSplitter


class TestEmbeddingBasedDocumentSplitter:
    def test_init(self):
        """Test initialization with valid parameters."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(
            text_embedder=mock_embedder,
            sentences_per_group=2,
            percentile=0.9,
            min_length=50,
            max_length=1000,
        )

        assert splitter.text_embedder == mock_embedder
        assert splitter.sentences_per_group == 2
        assert splitter.percentile == 0.9
        assert splitter.min_length == 50
        assert splitter.max_length == 1000

    def test_init_invalid_sentences_per_group(self):
        """Test initialization with invalid sentences_per_group."""
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="sentences_per_group must be greater than 0"):
            EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, sentences_per_group=0)

    def test_init_invalid_percentile(self):
        """Test initialization with invalid percentile."""
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="percentile must be between 0.0 and 1.0"):
            EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, percentile=1.5)

    def test_init_invalid_min_length(self):
        """Test initialization with invalid min_length."""
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="min_length must be greater than or equal to 0"):
            EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, min_length=-1)

    def test_init_invalid_max_length(self):
        """Test initialization with invalid max_length."""
        mock_embedder = Mock()
        with pytest.raises(ValueError, match="max_length must be greater than min_length"):
            EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, min_length=100, max_length=50)

    def test_warm_up(self):
        """Test warm_up method."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)

        with patch('haystack.components.preprocessors.embedding_based_document_splitter.nltk_imports') as mock_nltk:
            mock_nltk.check.return_value = None
            with patch('haystack.components.preprocessors.embedding_based_document_splitter.SentenceSplitter') as mock_splitter_class:
                mock_splitter = Mock()
                mock_splitter_class.return_value = mock_splitter

                splitter.warm_up()

                assert splitter.sentence_splitter == mock_splitter
                mock_splitter_class.assert_called_once()

    def test_run_not_warmed_up(self):
        """Test run method when not warmed up."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)

        with pytest.raises(RuntimeError, match="wasn't warmed up"):
            splitter.run(documents=[Document(content="test")])

    def test_run_invalid_input(self):
        """Test run method with invalid input."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)
        splitter.sentence_splitter = Mock()

        with pytest.raises(TypeError, match="expects a List of Documents"):
            splitter.run(documents="not a list")

    def test_run_document_with_none_content(self):
        """Test run method with document having None content."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)
        splitter.sentence_splitter = Mock()

        with pytest.raises(ValueError, match="content for document ID"):
            splitter.run(documents=[Document(content=None)])

    def test_run_empty_document(self):
        """Test run method with empty document."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)
        splitter.sentence_splitter = Mock()

        result = splitter.run(documents=[Document(content="")])
        assert result["documents"] == []

    def test_group_sentences_single(self):
        """Test grouping sentences with sentences_per_group=1."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, sentences_per_group=1)

        sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
        groups = splitter._group_sentences(sentences)

        assert groups == sentences

    def test_group_sentences_multiple(self):
        """Test grouping sentences with sentences_per_group=2."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, sentences_per_group=2)

        sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3.", "Sentence 4."]
        groups = splitter._group_sentences(sentences)

        assert groups == ["Sentence 1. Sentence 2.", "Sentence 3. Sentence 4."]

    def test_cosine_distance(self):
        """Test cosine distance calculation."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)

        # Test with identical vectors
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]
        distance = splitter._cosine_distance(embedding1, embedding2)
        assert distance == 0.0

        # Test with orthogonal vectors
        embedding1 = [1.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0]
        distance = splitter._cosine_distance(embedding1, embedding2)
        assert distance == 1.0

        # Test with zero vectors
        embedding1 = [0.0, 0.0, 0.0]
        embedding2 = [1.0, 0.0, 0.0]
        distance = splitter._cosine_distance(embedding1, embedding2)
        assert distance == 1.0

    def test_find_split_points_empty(self):
        """Test finding split points with empty embeddings."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)

        split_points = splitter._find_split_points([])
        assert split_points == []

        split_points = splitter._find_split_points([[1.0, 0.0]])
        assert split_points == []

    def test_find_split_points(self):
        """Test finding split points with embeddings."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, percentile=0.5)

        # Create embeddings where the second pair has high distance
        embeddings = [
            [1.0, 0.0, 0.0],  # Similar to next
            [0.9, 0.1, 0.0],  # Similar to previous
            [0.0, 1.0, 0.0],  # Very different from next
            [0.1, 0.9, 0.0],  # Similar to previous
        ]

        split_points = splitter._find_split_points(embeddings)
        # Should find a split point after the second embedding (index 2)
        assert 2 in split_points

    def test_create_splits_from_points(self):
        """Test creating splits from split points."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)

        sentence_groups = ["Group 1", "Group 2", "Group 3", "Group 4"]
        split_points = [2]  # Split after index 1

        splits = splitter._create_splits_from_points(sentence_groups, split_points)
        assert splits == ["Group 1 Group 2", "Group 3 Group 4"]

    def test_create_splits_from_points_no_points(self):
        """Test creating splits with no split points."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)

        sentence_groups = ["Group 1", "Group 2", "Group 3"]
        split_points = []

        splits = splitter._create_splits_from_points(sentence_groups, split_points)
        assert splits == ["Group 1 Group 2 Group 3"]

    def test_merge_small_splits(self):
        """Test merging small splits."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder, min_length=10)

        splits = ["Short", "Also short", "Long enough text", "Another short"]
        merged = splitter._merge_small_splits(splits)

        assert len(merged) == 3
        assert "Short Also short" in merged[0]
        assert "Long enough text" in merged[1]
        assert "Another short" in merged[2]

    def test_create_documents_from_splits(self):
        """Test creating Document objects from splits."""
        mock_embedder = Mock()
        splitter = EmbeddingBasedDocumentSplitter(text_embedder=mock_embedder)

        original_doc = Document(content="test", meta={"key": "value"})
        splits = ["Split 1", "Split 2"]

        documents = splitter._create_documents_from_splits(splits, original_doc)

        assert len(documents) == 2
        assert documents[0].content == "Split 1"
        assert documents[0].meta["source_id"] == original_doc.id
        assert documents[0].meta["split_id"] == 0
        assert documents[0].meta["key"] == "value"
        assert documents[1].content == "Split 2"
        assert documents[1].meta["split_id"] == 1

    def test_to_dict(self):
        """Test serialization to dictionary."""
        mock_embedder = Mock()
        mock_embedder.to_dict.return_value = {"type": "MockEmbedder"}

        splitter = EmbeddingBasedDocumentSplitter(
            text_embedder=mock_embedder,
            sentences_per_group=2,
            percentile=0.9,
            min_length=50,
            max_length=1000,
        )

        result = splitter.to_dict()

        assert "EmbeddingBasedDocumentSplitter" in result["type"]
        assert result["init_parameters"]["sentences_per_group"] == 2
        assert result["init_parameters"]["percentile"] == 0.9
        assert result["init_parameters"]["min_length"] == 50
        assert result["init_parameters"]["max_length"] == 1000
        assert "text_embedder" in result["init_parameters"]


    @pytest.mark.integration
    def test_embedding_based_document_splitter_integration(self):
        """Integration test using real SentenceTransformersTextEmbedder."""
        # Use a lightweight model for speed
        embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        embedder.warm_up()

        splitter = EmbeddingBasedDocumentSplitter(
            text_embedder=embedder,
            sentences_per_group=2,
            percentile=0.9,
            min_length=30,
            max_length=300,
        )
        splitter.warm_up()

        # A document with multiple topics
        text = (
            "The weather today is beautiful. The sun is shining brightly. The temperature is perfect for a walk. "
            "Machine learning has revolutionized many industries. Neural networks can process vast amounts of data. "
            "Deep learning models achieve remarkable accuracy on complex tasks. "
            "Cooking is both an art and a science. Fresh ingredients make all the difference. "
            "Proper seasoning enhances the natural flavors of food. "
            "The history of ancient civilizations fascinates researchers. Archaeological discoveries reveal new insights. "
            "Ancient texts provide valuable information about past societies."
        )
        doc = Document(content=text)

        result = splitter.run(documents=[doc])
        split_docs = result["documents"]

        # There should be more than one split
        assert len(split_docs) > 1
        # Each split should be non-empty and respect min_length
        for split_doc in split_docs:
            assert split_doc.content.strip() != ""
            assert len(split_doc.content) >= 30
        # The splits should cover the original text
        combined = " ".join([d.content for d in split_docs]).replace(" ", "")
        original = text.replace(" ", "")
        assert combined in original or original in combined
