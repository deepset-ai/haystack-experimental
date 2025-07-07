# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Protocol

from haystack import Document

# See https://github.com/pylint-dev/pylint/issues/9319.
# pylint: disable=unnecessary-ellipsis


class DocumentEmbedder(Protocol):
    """
    Protocol for Document Embedders.
    """

    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate embeddings for the input documents.

        Implementing classes may accept additional optional parameters in their run method.
        For example: `def run (self, documents: List[Document], param_a="default", param_b="another_default")`.

        :param documents:
            The input documents to be embedded.
        :returns:
            A dictionary containing the keys:
                - 'documents', which is expected to be a List[Document] with embeddings added to each document.
                - any optional keys such as 'metadata'.
        """
        ...
