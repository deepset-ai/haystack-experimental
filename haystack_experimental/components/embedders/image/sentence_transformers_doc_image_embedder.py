# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackend,
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.dataclasses.byte_stream import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.device import ComponentDevice
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs
from typing_extensions import NotRequired

from haystack_experimental.components.image_converters.image_utils import (
    _convert_pdf_to_pil_images,
)
from haystack_experimental.dataclasses.image_content import IMAGE_MIME_TYPES

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image as PILImage
    from PIL.Image import Image
    from PIL.ImageFile import ImageFile


class _ImageSourceInfo(TypedDict):
    path: Path
    type: Literal["image", "pdf"]
    page_number: NotRequired[int]  # Only present for PDF documents


class _PdfPageInfo(TypedDict):
    doc_idx: int
    path: Path
    page_number: int


@component
class SentenceTransformersDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Sentence Transformers models.

    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    def __init__(
        self,
        *,
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        model: str = "sentence-transformers/clip-ViT-B-32",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        batch_size: int = 32,
        progress_bar: bool = True,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        encode_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        """
        Creates a SentenceTransformersDocumentEmbedder component.

        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param model:
            The Sentence Transformers model to use for calculating embeddings. To be used with this component,
            the model must be able to embed images and text into the same vector space.
            Pass a local path or ID of the model on Hugging Face.
        :param device:
            The device to use for loading the model.
            Overrides the default device.
        :param token:
            The API token to download private models from Hugging Face.
        :param batch_size:
            Number of documents to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar when embedding documents.
        :param trust_remote_code:
            If `False`, allows only Hugging Face verified model architectures.
            If `True`, allows custom models and scripts.
        :param local_files_only:
            If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
        :param model_kwargs:
            Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
        :param tokenizer_kwargs:
            Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
            Refer to specific model documentation for available kwargs.
        :param config_kwargs:
            Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
        :param precision:
            The precision to use for the embeddings.
            All non-float32 precisions are quantized embeddings.
            Quantized embeddings are smaller and faster to compute, but may have a lower accuracy.
            They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
        :param encode_kwargs:
            Additional keyword arguments for `SentenceTransformer.encode` when embedding documents.
            This parameter is provided for fine customization. Be careful not to clash with already set parameters and
            avoid passing parameters that change the output type.
        :param backend:
            The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
            Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
            for more information on acceleration and quantization options.
        """
        pillow_import.check()

        self.file_path_meta_field = file_path_meta_field
        self.root_path = root_path or ""
        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.encode_kwargs = encode_kwargs
        self.precision = precision
        self.backend = backend
        self._embedding_backend: Optional[_SentenceTransformersEmbeddingBackend] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
            model=self.model,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            precision=self.precision,
            encode_kwargs=self.encode_kwargs,
            backend=self.backend,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersDocumentImageEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        deserialize_secrets_inplace(init_params, keys=["token"])
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Initializes the component.
        """
        if self._embedding_backend is None:
            self._embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model=self.model,
                device=self.device.to_torch_str(),
                auth_token=self.token,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                backend=self.backend,
            )
            if self.tokenizer_kwargs and self.tokenizer_kwargs.get("model_max_length"):
                self._embedding_backend.model.max_seq_length = self.tokenizer_kwargs["model_max_length"]

    @staticmethod
    def _extract_image_sources_info(
        documents: List[Document], file_path_meta_field: str, root_path: str
    ) -> List[_ImageSourceInfo]:
        """
        Extracts the image source information from the documents.

        :param documents: List of documents to extract image source information from.
        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located.

        :returns:
            A list of _ImageSourceInfo dictionaries, each containing the path and type of the image.
            If the image is a PDF, the dictionary also contains the page number.
        :raises ValueError: If the document is missing the file_path_meta_field key in its metadata, the file path is
            invalid, the MIME type is not supported, or the page number is missing for a PDF document.
        """
        images_source_info: List[_ImageSourceInfo] = []
        for doc in documents:
            file_path = doc.meta.get(file_path_meta_field)
            if file_path is None:
                raise ValueError(
                    f"Document with ID '{doc.id}' is missing the '{file_path_meta_field}' key in its metadata."
                    f" Please ensure that the documents you are trying to convert have this key set."
                )

            resolved_file_path = Path(root_path, file_path)
            if not resolved_file_path.is_file():
                raise ValueError(
                    f"Document with ID '{doc.id}' has an invalid file path '{resolved_file_path}'. "
                    f"Please ensure that the documents you are trying to convert have valid file paths."
                )

            mime_type = doc.meta.get("mime_type") or mimetypes.guess_type(resolved_file_path)[0]
            if mime_type not in IMAGE_MIME_TYPES:
                raise ValueError(
                    f"Document with file path '{resolved_file_path}' has an unsupported MIME type '{mime_type}'. "
                    f"Please ensure that the documents you are trying to convert are of the supported "
                    f"types: {', '.join(IMAGE_MIME_TYPES)}."
                )

            # If mimetype is PDF we also need the page number to be able to convert the right page
            if mime_type == "application/pdf":
                page_number = doc.meta.get("page_number")
                if page_number is None:
                    raise ValueError(
                        f"Document with ID '{doc.id}' comes from the PDF file '{resolved_file_path}' but is missing "
                        f"the 'page_number' key in its metadata. Please ensure that PDF documents you are trying to "
                        f"convert have this key set."
                    )
                pdf_info: _ImageSourceInfo = {"path": resolved_file_path, "type": "pdf", "page_number": page_number}
                images_source_info.append(pdf_info)
            else:
                image_info: _ImageSourceInfo = {"path": resolved_file_path, "type": "image"}
                images_source_info.append(image_info)

        return images_source_info

    @staticmethod
    def _process_pdf_files(
        pdf_pages_info: List[_PdfPageInfo],
        size: Optional[Tuple[int, int]] = None,
    ) -> Dict[int, "Image"]:
        """
        Process PDF files and return a mapping of document indices to converted PIL images.

        :param pdf_pages_info: List of _PdfPageInfo dictionaries with doc_idx, path, and page_number.
        :param size: Optional tuple of width and height to resize the images to.
        :returns: Dictionary mapping document indices to PIL images.
        """
        if not pdf_pages_info:
            return {}

        pdf_images_by_doc_idx = {}
        pdf_files_by_path = defaultdict(list)

        for pdf_page_info in pdf_pages_info:
            pdf_files_by_path[pdf_page_info["path"]].append((pdf_page_info["doc_idx"], pdf_page_info["page_number"]))

        # Open and convert each PDF file once
        for file_path, doc_page_pairs in pdf_files_by_path.items():
            page_numbers = [page_num for _, page_num in doc_page_pairs]
            bytestream = ByteStream.from_file_path(file_path)
            pdf_images = _convert_pdf_to_pil_images(bytestream=bytestream, page_range=page_numbers, size=size)
            # Map back to document positions
            page_to_pil_image = dict(pdf_images)
            for doc_idx, page_num in doc_page_pairs:
                pdf_images_by_doc_idx[doc_idx] = page_to_pil_image[page_num]

        return pdf_images_by_doc_idx

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Embed a list of documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "SentenceTransformersDocumentImageEmbedder expects a list of Documents as input. "
                "In case you want to embed a list of strings, please use the SentenceTransformersTextEmbedder."
            )
        if self._embedding_backend is None:
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        images_source_info = self._extract_image_sources_info(
            documents=documents,
            file_path_meta_field=self.file_path_meta_field,
            root_path=self.root_path,
        )

        images_to_embed: List = [None] * len(documents)
        pdf_pages_info: List[_PdfPageInfo] = []

        for doc_idx, image_source_info in enumerate(images_source_info):
            if image_source_info["type"] == "image":
                # Process images directly
                image: Union["Image", "ImageFile"] = PILImage.open(image_source_info["path"])
                images_to_embed[doc_idx] = image
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

        # Process PDF files and update images_to_embed
        pdf_images = self._process_pdf_files(pdf_pages_info=pdf_pages_info)
        for doc_idx, pil_image in pdf_images.items():
            images_to_embed[doc_idx] = pil_image

        embeddings = self._embedding_backend.embed(
            # TODO: when moving this component to Haystack, adjust the signature of the embedding backend embed method
            # to also accept a list of PIL images
            images_to_embed,  # type: ignore[arg-type]
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            precision=self.precision,
            **(self.encode_kwargs if self.encode_kwargs else {}),
        )

        docs_with_embeddings = []
        for doc, emb in zip(documents, embeddings):
            copied_doc = copy(doc)
            copied_doc.embedding = emb
            # we store this information for later inspection
            copied_doc.meta["embedding_source"] = {"type": "image", "file_path_meta_field": self.file_path_meta_field}
            docs_with_embeddings.append(copied_doc)

        return {"documents": docs_with_embeddings}
