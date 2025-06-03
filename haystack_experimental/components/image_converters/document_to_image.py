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
    A component to convert documents sourced from PDF, and image files to ImageContent objects.

    Documents will be skipped under the following scenarios:
    - If the `file_path` is not present in the metadata.
    - If the file path does not start with the expected root path.
    - If the file is not one of the supported image formats.
    - If the `page_number` is not present in the metadata if the file path points to a PDF file.
    """

    def __init__(
        self,
        *,
        root_path: Optional[str] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None
    ):
        """
        Create the DocumentToImageContent component.

        :param root_path: The root path of the document to convert.
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

        This de-duplication is relevant for PDF documents where the same file path and page number combination can
        occur because the same PDF page could have produced multiple documents, e.g., when splitting.
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
        Convert documents sourced from PDF files to base64 encoded images.

        :param documents: A list of documents with image information in their metadata.
        :returns:
            A list of text documents and a corresponding list of base64 encoded images.
        """
        if not documents:
            return {"image_documents": [], "image_contents": [], "non_image_documents": []}

        pdf_docs = []
        image_docs = []
        non_image_docs = []
        missing_info_docs = []

        for doc in documents:
            file_path = doc.meta.get("file_path")
            if file_path is None:
                missing_info_docs.append(doc)
                continue

            # Store the non-image documents separately
            mime_type = doc.meta.get("mime_type") or mimetypes.guess_type(file_path)[0]
            if mime_type not in IMAGE_MIME_TYPES:
                non_image_docs.append(doc)

            # If mimetype is PDF we also need the page number to be able to convert the right page
            if mime_type == "application/pdf":
                page_number = doc.meta.get("page_number")
                if page_number is None:
                    missing_info_docs.append(doc)
                    continue
                pdf_docs.append(doc)
            else:
                image_docs.append(doc)

        if missing_info_docs:
            logger.warning(
                "Calling DocumentToImageContent with {len_missing_file_paths} out of {len_docs} Documents that "
                "are missing file paths in their metadata.",
                len_missing_file_paths=len(missing_info_docs),
                len_docs=len(documents),
            )

        # We de-duplicate the pdf documents because it's possible that the same PDF page is represented by multiple
        # documents
        pdf_docs = self._deduplicate(pdf_docs)

        # Convert the image documents into ImageContent objects via ByteStream
        image_byte_streams: List[Union[str, Path, ByteStream]] = [
            ByteStream.from_file_path(
                filepath=doc.meta["file_path"],
                mime_type=mimetypes.guess_type(doc.meta["file_path"])[0],
                meta={"file_path": doc.meta["file_path"]}
            ) for doc in image_docs
        ]
        image_contents = self._file_to_image_converter.run(sources=image_byte_streams)["image_contents"]

        # Convert the PDF documents into ImageContent objects via ByteStream
        pdf_to_image_inputs = {
            "sources": [
                ByteStream.from_file_path(
                    filepath=Path(doc.meta["file_path"]),
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
