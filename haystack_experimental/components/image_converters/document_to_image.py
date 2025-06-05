# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from haystack import Document, component, logging
from haystack.dataclasses import ByteStream

from haystack_experimental.components.image_converters.file_to_image import ImageFileToImageContent
from haystack_experimental.components.image_converters.pdf_to_image import PDFToImageContent
from haystack_experimental.dataclasses.image_content import IMAGE_MIME_TYPES, ImageContent

logger = logging.getLogger(__name__)


@component
class DocumentToImageContent:
    """
    Converts documents sourced from PDF and image files into ImageContents.

    This component processes a list of documents and extracts visual content from supported file formats, converting
    them into ImageContents that can be used for multimodal AI tasks. It handles both direct image files and PDF
    documents by extracting specific pages as images.

    The component categorizes input documents into three groups:
    - Image documents: Successfully converted to ImageContent objects
    - Non-image documents: Documents that don't contain supported image formats
    - Documents with missing information: Documents lacking required metadata such as `file_path` or `page_number`.

    Documents will be skipped and categorized as non-image documents in the following scenarios:
    - The `file_path` is not present in the document metadata
    - The file path does not point to an existing file when combined with `root_path`
    - The file format is not among the supported image types
    - For PDF files, the `page_number` is not present in the metadata

    Example:
        ```python
        converter = DocumentToImageContent(
            root_path="/data/documents",
            detail="high",
            size=(800, 600)
        )

        documents = [
            Document(content="Optional description of image.jpg", meta={"file_path": "image.jpg"}),
            Document(content="Text content of page 1 of doc.pdf", meta={"file_path": "doc.pdf", "page_number": 1})
        ]

        result = converter.run(documents)
        image_documents = result["image_documents"]
        image_contents = result["image_contents"]
        ```
    """

    def __init__(
        self,
        *,
        root_path: Optional[str] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the DocumentToImageContent component.

        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param detail: Optional detail level of the image (only supported by OpenAI). One of "auto", "high", or "low".
            This will be passed to the created ImageContent objects.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        """
        self.root_path = root_path
        self.detail = detail
        self.size = size

        # Initializing these converters will trigger the lazy import checks.
        self._file_to_image_converter = ImageFileToImageContent(detail=detail, size=size)
        self._pdf_to_image_converter = PDFToImageContent(detail=detail, size=size)

    @staticmethod
    def _deduplicate(documents: List[Document]) -> List[Document]:
        """
        Deduplicate the documents based on file path and page number if available.

        This deduplication is particularly important for PDF documents where the same page might be represented by
        multiple Documents due to text splitting or other preprocessing steps. Only the first occurrence
        of each unique (file_path, page_number) combination is retained.

        :param documents: List of Documents to deduplicate.

        :returns:
            List of Documents with duplicates removed, maintaining original order of first occurrences.
        """
        unique_documents = []
        seen = set()
        for doc in documents:
            key = (doc.meta["file_path"], doc.meta.get("page_number"))
            if key not in seen:
                unique_documents.append(doc)
                seen.add(key)
        return unique_documents

    @component.output_types(
        image_documents=List[Document], image_contents=List[ImageContent], non_image_documents=List[Document]
    )
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Convert documents with image or PDF sources to ImageContents.

        This method processes the input documents, extracting images from supported file formats and converting them
        to ImageContents. Documents are categorized based on their file type and processing success.

        :param documents: List of Documents to process. Each document should have metadata containing at minimum
            a 'file_path' key. PDF documents additionally require a 'page_number' key to specify which page to convert.

        :returns:
            Dictionary containing three lists:
            - "image_documents": Document objects that were successfully processed and have corresponding ImageContents.
                Includes both image files and PDF pages.
            - "image_contents": ImageContents created from the processed documents. These contain base64-encoded image
                data and metadata. The order corresponds to the `image_documents` list.
            - "non_image_documents": Document objects that could not be processed as images. This includes unsupported
                file types, missing files, and documents with insufficient metadata.
        """
        if not documents:
            return {"image_documents": [], "image_contents": [], "non_image_documents": []}

        root_path = self.root_path or ""

        pdf_docs = []
        image_docs = []
        non_image_docs = []
        missing_info_docs = []

        for doc in documents:
            file_path = doc.meta.get("file_path")
            if file_path is None or not Path(root_path, file_path).is_file():
                missing_info_docs.append(doc)
                continue

            # Store the non-image documents separately
            mime_type = doc.meta.get("mime_type") or mimetypes.guess_type(file_path)[0]
            if mime_type not in IMAGE_MIME_TYPES:
                non_image_docs.append(doc)
                continue

            # If mimetype is PDF we also need the page number to be able to convert the right page
            if mime_type == "application/pdf":
                page_number = doc.meta.get("page_number")
                if page_number is None:
                    missing_info_docs.append(doc)
                else:
                    pdf_docs.append(doc)
                continue

            image_docs.append(doc)

        if missing_info_docs:
            logger.warning(
                "In DocumentToImageContent {len_missing_file_paths} Documents are either missing a `file_path` "
                "key in their metadata or the `file_path` does not point to a valid file. "
                "They will be returned in the `non_image_documents` output.",
                len_missing_file_paths=len(missing_info_docs),
            )

        # We de-duplicate the pdf documents because it's possible that the same PDF page is represented by multiple
        # documents
        pdf_docs = self._deduplicate(pdf_docs)

        # Convert the image documents into ImageContent objects
        image_byte_streams: List[Union[str, Path, ByteStream]] = [
            ByteStream.from_file_path(
                filepath=Path(root_path, doc.meta["file_path"]),
                mime_type=mimetypes.guess_type(doc.meta["file_path"])[0],
                meta={"file_path": doc.meta["file_path"]}
            ) for doc in image_docs
        ]
        image_contents = self._file_to_image_converter.run(sources=image_byte_streams)["image_contents"]

        # Convert the PDF documents into ImageContent objects
        pdf_to_image_inputs = {
            "sources": [
                ByteStream.from_file_path(
                    filepath=Path(root_path, doc.meta["file_path"]),
                    mime_type="application/pdf",
                    meta={"page_number": doc.meta["page_number"], "file_path": doc.meta["file_path"]}
                ) for doc in pdf_docs
            ],
            "page_range": [doc.meta["page_number"] for doc in pdf_docs],
        }
        pdf_image_contents = self._pdf_to_image_converter.run(
            sources=pdf_to_image_inputs["sources"],
            page_range=pdf_to_image_inputs["page_range"],
        )["image_contents"]

        return {
            "image_documents": image_docs + pdf_docs,
            "image_contents": image_contents + pdf_image_contents,
            "non_image_documents": non_image_docs + missing_info_docs
        }
