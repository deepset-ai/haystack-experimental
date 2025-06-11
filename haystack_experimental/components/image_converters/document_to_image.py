# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from haystack import Document, component, logging
from haystack.dataclasses import ByteStream

from haystack_experimental.components.image_converters.file_to_image import ImageFileToImageContent
from haystack_experimental.components.image_converters.image_utils import _extract_image_sources_info, _PdfPageInfo
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
    - The `file_path_meta_field` key with a valid file path that exists when combined with `root_path`
    - A supported image format (MIME type must be one of the supported image types)
    - For PDF files, a `page_number` key specifying which page to extract

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
        size: Optional[Tuple[int, int]] = None,
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

        images_source_info = _extract_image_sources_info(
            documents=documents,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
        )

        image_contents = []

        pdf_pages_info: List[_PdfPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["type"] == "image":
                # Process images directly
                image_contents.extend(
                    self._file_to_image_converter.run(
                        sources=[
                            ByteStream.from_file_path(
                                filepath=image_source_info["path"],
                                mime_type=image_source_info["mime_type"],
                                meta={"file_path": documents[doc_idx].meta[self.file_path_meta_field]},
                            )
                        ]
                    )["image_contents"]
                )
            else:
                # Store PDF documents for later processing
                page_number = image_source_info.get("page_number")
                assert page_number is not None  # checked in _extract_image_sources_info but mypy doesn't know that
                pdf_page_info: _PdfPageInfo = {
                    "doc_idx": doc_idx,
                    "path": image_source_info["path"],
                    "page_number": page_number,
                }
                pdf_pages_info.append(pdf_page_info)

        pdf_files_by_path = defaultdict(list)

        for pdf_page_info in pdf_pages_info:
            pdf_files_by_path[pdf_page_info["path"]].append((pdf_page_info["doc_idx"], pdf_page_info["page_number"]))

        # Open and convert each PDF file once
        for file_path, doc_page_pairs in pdf_files_by_path.items():
            page_numbers = [page_num for _, page_num in doc_page_pairs]
            meta = {"file_path": documents[doc_idx].meta[self.file_path_meta_field]}
            # "page_number": page_numbers}
            bytestream = ByteStream.from_file_path(filepath=file_path, mime_type="application/pdf", meta=meta)

            image_contents.extend(
                # Possible for _pdf_to_image_converter to return multiple images depending on the page range
                self._pdf_to_image_converter.run(
                    sources=[bytestream],
                    page_range=page_numbers,
                )["image_contents"]
            )

        return {"image_contents": image_contents}
