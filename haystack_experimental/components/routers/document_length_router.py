# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from haystack import component
from haystack.dataclasses import Document


@component
class DocumentLengthRouter:
    """
    Categorizes documents by length based on their content and routes them to the appropriate output.

    A common use case for DocumentLengthRouter is handling documents obtained from PDFs that contain non-text
    content, such as scanned pages or images. This component can detect empty or low-content documents and route them to
    components that perform OCR, generate captions or compute image embeddings.

    ### Usage example

    ```python
    from haystack_experimental.components.routers import DocumentLengthRouter
    from haystack.dataclasses import Document

    docs = [
        Document(content="Short"),
        Document(content="Long document "*20),
    ]

    router = DocumentLengthRouter(threshold=10)

    result = router.run(documents=docs)
    print(result)

    # {
    #     "short_documents": [Document(content="Short", ...)],
    #     "long_documents": [Document(content="Long document ...", ...)],
    # }
    """

    def __init__(
        self,
        *,
        threshold: int = 10,
    ) -> None:
        """
        Initialize the DocumentLengthRouter component.

        :param threshold:
            The threshold for the length of the document content. Documents where the content is None or the length of
            the content is less than or equal to the threshold will be routed to the short_documents output. Otherwise,
            the document will be routed to the long_documents output.
        """
        self.threshold = threshold

    @component.output_types(
        short_documents=List[Document],
        long_documents=List[Document],
    )
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Categorize input documents into groups based on their length.

        :param documents:
            A list of documents to be categorized.

        :returns: A dictionary with the following keys:
            - `short_documents`: A list of documents where content is None or the length of the content is less than or
               equal to the threshold.
            - `long_documents`: A list of documents where the length of the content is greater than the threshold.
        """
        short_documents = []
        long_documents = []

        for doc in documents:
            if doc.content is None or len(doc.content) <= self.threshold:
                short_documents.append(doc)
            else:
                long_documents.append(doc)

        return {"short_documents": short_documents, "long_documents": long_documents}
