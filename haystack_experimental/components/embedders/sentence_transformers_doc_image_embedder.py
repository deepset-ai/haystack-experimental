# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackend,
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

with LazyImport("Run 'pip install pillow'") as pillow_import:
    from PIL import Image


@component
class SentenceTransformersDocumentImageEmbedder:
    """
    A component for computing Document embeddings based on images using Sentence Transformers models.

    The embedding of each Document is stored in the `embedding` field of the Document.
    """

    def __init__(  # noqa: PLR0913 # pylint: disable=too-many-positional-arguments
        self,
        *,
        meta_field_for_image_path: str = "image_path",
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

        :param meta_field_for_image_path:
            The field in the Document metadata that contains the path to the image.
        :param model:
            The model to use for calculating embeddings.
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

        self.meta_field_for_image_path = meta_field_for_image_path
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
        self.embedding_backend: Optional[_SentenceTransformersEmbeddingBackend] = None
        self.precision = precision
        self.backend = backend

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
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
        if self.embedding_backend is None:
            self.embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
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
                self.embedding_backend.model.max_seq_length = self.tokenizer_kwargs["model_max_length"]

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
                "SentenceTransformersDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the SentenceTransformersTextEmbedder."
            )
        if self.embedding_backend is None:
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        images_to_embed = []
        for doc in documents:
            image_path = doc.meta.get(self.meta_field_for_image_path)
            if not image_path:
                raise ValueError(f"Image path not found in document metadata for document {doc.id}")
            images_to_embed.append(Image.open(image_path))

        embeddings = self.embedding_backend.embed(
            images_to_embed,
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            precision=self.precision,
            **(self.encode_kwargs if self.encode_kwargs else {}),
        )

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            doc.meta["embedding_source"] = f"meta.{self.meta_field_for_image_path}"

        return {"documents": documents}
