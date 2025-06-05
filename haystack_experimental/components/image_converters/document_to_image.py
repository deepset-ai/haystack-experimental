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

    Documents are expected to have metadata containing:
    - The `file_path_meta_field` is not present in the document metadata
    - The file path does not point to an existing file when combined with `root_path`
    - The file format is not among the supported image types
    - For PDF files, the `page_number` is not present in the metadata
    Otherwise, the component will raise a `ValueError` with a descriptive message.

    Usage example:
        ```python
        from haystack import Document
        from haystack_experimental.components.image_converters.document_to_image import DocumentToImageContent

        converter = DocumentToImageContent(
            file_path_meta_field="file_path",
            root_path="/data/documents",
            detail="high",
            size=(800, 600)
        )

        documents = [
            Document(content="Optional description of image.jpg", meta={"file_path": "image.jpg"}),
            Document(content="Text content of page 1 of doc.pdf", meta={"file_path": "doc.pdf", "page_number": 1})
        ]

        result = converter.run(documents)
        image_contents = result["image_contents"]
        # [ImageContent(
        #    base64_image='/9j/4A...', mime_type='image/jpeg', detail='high', meta={'file_path': 'image.jpg'}
        #  ),
        #  ImageContent(
        #    base64_image='/9j/4A...', mime_type='image/jpeg', detail='high',
        #    meta={'page_number': 1, 'file_path': 'doc.pdf'}
        #  )]
        ```
    """

    def __init__(
        self,
        *,
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        detail: Optional[Literal["auto", "high", "low"]] = None,
        size: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the DocumentToImageContent component.

        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param detail: Optional detail level of the image (only supported by OpenAI). Can be "auto", "high", or "low".
            This will be passed to the created ImageContent objects.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
        """
        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.detail = detail
        self.size = size

        # Initializing these converters will trigger the lazy import checks.
        self._file_to_image_converter = ImageFileToImageContent(detail=detail, size=size)
        self._pdf_to_image_converter = PDFToImageContent(detail=detail, size=size)

    @component.output_types(image_contents=List[ImageContent])
    def run(self, documents: List[Document]) -> Union[Dict[str, List[ImageContent]], Dict[str, List]]:
        """
        Convert documents with image or PDF sources into ImageContent objects.

        This method processes the input documents, extracting images from supported file formats and converting them
        into ImageContent objects.

        :param documents: A list of documents to process. Each document should have metadata containing at minimum
            a 'file_path_meta_field' key. PDF documents additionally require a 'page_number' key to specify which
            page to convert.

        :returns:
            Dictionary containing one key:
            - "image_contents": ImageContents created from the processed documents. These contain base64-encoded image
                data and metadata. The order corresponds to order of input documents.
        :raises ValueError:
            If any document is missing the required metadata keys, has an invalid file path, or has an unsupported
            MIME type. The error message will specify which document and what information is missing or incorrect.
        """
        if not documents:
            return {"image_contents": []}

        pdf_docs = []
        image_docs = []

        for doc in documents:
            file_path = doc.meta.get(self.file_path_meta_field)
            if file_path is None:
                raise ValueError(
                    f"Document with ID '{doc.id}' is missing the '{self.file_path_meta_field}' key in its metadata."
                    f" Please ensure that the documents you are trying to convert have this key set."
                )

            if not Path(self.root_path, file_path).is_file():
                raise ValueError(
                    f"Document with ID '{doc.id}' has an invalid file path '{file_path}'. "
                    f"Please ensure that the documents you are trying to convert have valid file paths."
                )

            mime_type = doc.meta.get("mime_type") or mimetypes.guess_type(file_path)[0]
            if mime_type not in IMAGE_MIME_TYPES:
                raise ValueError(
                    f"Document with file path '{file_path}' has an unsupported MIME type '{mime_type}'. "
                    f"Please ensure that the documents you are trying to convert are of the supported "
                    f"types: {', '.join(IMAGE_MIME_TYPES)}."
                )

            # If mimetype is PDF we also need the page number to be able to convert the right page
            if mime_type == "application/pdf":
                page_number = doc.meta.get("page_number")
                if page_number is None:
                    raise ValueError(
                        f"Document with ID '{doc.id}' comes from the PDF file '{file_path}' but is missing the "
                        f"'page_number' key in its metadata. Please ensure that PDF documents you are trying to "
                        f"convert have this key set."
                    )
                pdf_docs.append(doc)
            else:
                image_docs.append(doc)

        # Convert the image documents into ImageContent objects
        image_byte_streams: List[Union[str, Path, ByteStream]] = [
            ByteStream.from_file_path(
                filepath=Path(self.root_path, doc.meta["file_path"]),
                mime_type=mimetypes.guess_type(doc.meta["file_path"])[0],
                meta={"file_path": doc.meta["file_path"]}
            ) for doc in image_docs
        ]
        image_contents = self._file_to_image_converter.run(sources=image_byte_streams)["image_contents"]

        # Convert the PDF documents into ImageContent objects
        pdf_to_image_inputs = {
            "sources": [
                ByteStream.from_file_path(
                    filepath=Path(self.root_path, doc.meta["file_path"]),
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

        return {"image_contents": image_contents + pdf_image_contents}
