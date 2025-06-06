# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackend,
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.dataclasses import ByteStream
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

from haystack_experimental.components.image_converters.image_utils import (
    _convert_pdf_to_pil_images,
    _resize_image_preserving_aspect_ratio,
)
from haystack_experimental.dataclasses.image_content import IMAGE_MIME_TYPES

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image as PILImage
    from PIL.Image import Image
    from PIL.ImageFile import ImageFile


@component
class SentenceTransformersDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Sentence Transformers models.

    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    def __init__(  # noqa: PLR0913 # pylint: disable=too-many-positional-arguments
        self,
        *,
        file_path_meta_field: str = "file_path",
        root_path: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
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
    ):
        """
        Creates a SentenceTransformersDocumentEmbedder component.

        :param file_path_meta_field: The metadata field in the Document that contains the file path to the image or PDF.
        :param root_path: The root directory path where document files are located. If provided, file paths in
            document metadata will be resolved relative to this path. If None, file paths are treated as absolute paths.
        :param size: If provided, resizes the image to fit within the specified dimensions (width, height) while
            maintaining aspect ratio. This reduces file size, memory usage, and processing time, which is beneficial
            when working with models that have resolution constraints or when transmitting images to remote services.
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
        :param truncate_dim:
            The dimension to truncate sentence embeddings to. `None` does no truncation.
            If the model wasn't trained with Matryoshka Representation Learning,
            truncating embeddings can significantly affect performance.
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
        self.size = size
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
            size=self.size,
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

    def warm_up(self):
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

    def _validate_image_paths(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Validates the image paths in the documents.

        :returns:
            A list of dictionaries, each dictionary containing the path, type, and page number of the image (for PDF).
        """
        images_paths_types = []
        for doc in documents:
            file_path = doc.meta.get(self.file_path_meta_field)
            if file_path is None:
                raise ValueError(
                    f"Document with ID '{doc.id}' is missing the '{self.file_path_meta_field}' key in its metadata."
                    f" Please ensure that the documents you are trying to convert have this key set."
                )

            resolved_file_path = Path(self.root_path, file_path)
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
                images_paths_types.append({"path": resolved_file_path, "type": "pdf", "page_number": page_number})
            else:
                images_paths_types.append({"path": resolved_file_path, "type": "image"})

        return images_paths_types

    @staticmethod
    def _process_pdf_documents(
        images_to_embed: Union[List["Image"], List["ImageFile"], List[None]],
        pdf_documents: List[Dict[str, Any]],
        size: Optional[Tuple[int, int]],
    ):
        """
        Process PDF documents and populate the images_to_embed list with converted images.

        :param images_to_embed: List to populate with converted PIL images (modified in place).
        :param pdf_documents: List of dictionaries with doc_idx, path, and page_number.
        :param size: Optional tuple of width and height to resize the images to.
        """
        if not pdf_documents:
            return

        pdf_files_by_path = defaultdict(list)
        for pdf_doc in pdf_documents:
            pdf_files_by_path[pdf_doc["path"]].append((pdf_doc["doc_idx"], pdf_doc["page_number"]))

        # Open and convert each PDF file once
        for file_path, doc_page_pairs in pdf_files_by_path.items():
            page_numbers = [page_num for _, page_num in doc_page_pairs]
            bytestream = ByteStream.from_file_path(file_path)
            pdf_images = _convert_pdf_to_pil_images(bytestream=bytestream, page_range=page_numbers, size=size)
            # Map back to document positions
            page_to_pil_image = dict(pdf_images)
            for doc_idx, page_num in doc_page_pairs:
                images_to_embed[doc_idx] = page_to_pil_image[page_num]

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
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

        images_paths_types = self._validate_image_paths(documents)

        images_to_embed: List[Union["Image", "ImageFile", None]] = [None] * len(documents)
        pdf_documents = []

        for doc_idx, image_path_type in enumerate(images_paths_types):
            if image_path_type["type"] == "image":
                # Process images directly
                image: Union["Image", "ImageFile"] = PILImage.open(image_path_type["path"])
                if self.size is not None:
                    image = _resize_image_preserving_aspect_ratio(image, self.size)
                images_to_embed[doc_idx] = image
            else:
                # Store PDF documents for later processing
                pdf_documents.append(
                    {"doc_idx": doc_idx, "path": image_path_type["path"], "page_number": image_path_type["page_number"]}
                )

        # Process PDF files
        self._process_pdf_documents(images_to_embed=images_to_embed, pdf_documents=pdf_documents, size=self.size)

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
            copied_doc.meta["embedding_source_type"] = "image"
            copied_doc.meta["embedding_source_file_path_meta_field"] = f"meta.{self.file_path_meta_field}"
            docs_with_embeddings.append(copied_doc)

        return {"documents": docs_with_embeddings}
