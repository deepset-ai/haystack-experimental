# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Protocol, TypeVar

# Ellipsis are needed to define the Protocol but pylint complains. See https://github.com/pylint-dev/pylint/issues/9319.
# pylint: disable=unnecessary-ellipsis

KeywordRetrieverT = TypeVar("KeywordRetrieverT", bound="KeywordRetriever")
EmbeddingRetrieverT = TypeVar("EmbeddingRetrieverT", bound="EmbeddingRetriever")


class KeywordRetriever(Protocol):
    """
    This protocol defines the minimal interface that all keyword-based BM25 Retrievers must implement.

    Retrievers are components that process a query and, based on that query, return relevant documents from a document
    store or other data source. They return a dictionary with a list of Document objects.
    """

    def run(self, query: str, filters: Optional[dict[str, Any]] = None, top_k: Optional[int] = None) -> dict[str, Any]:
        """
        Retrieve documents that are relevant to the query.

        Implementing classes may accept additional optional parameters in their run method.

        :param query:
        :param filters:
        :param top_k:
        :return:
        """
        ...


class EmbeddingRetriever(Protocol):
    """
    This protocol defines the minimal interface that all embedding-based Retrievers must implement.

    Retrievers are components that process a query and, based on that query, return relevant documents from a document
    store or other data source. They return a dictionary with a list of Document objects.
    """

    def run(
        self, query_embeddings: list[float], filters: Optional[dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Retrieve documents that are relevant to the query.

        Implementing classes may accept additional optional parameters in their run method.

        :param query_embeddings:
        :param filters:
        :param top_k:
        :return:
        """
        ...
