# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from cohere import ClientV2
from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack_integrations.components.embedders.cohere.embedding_types import EmbeddingTypes

from haystack_experimental.dataclasses.image_content import ImageContent


@component
class CohereDocumentImageEmbedder:
    """
    A component for computing Document embeddings using Cohere models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from cohere_haystack.embedders.document_embedder import CohereDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = CohereDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [-0.453125, 1.2236328, 2.0058594, ...]
    ```
    """

    def __init__(
        self,
        *,
        meta_field_for_image_path: str = "image_path",
        api_key: Secret = Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"]),
        model: str = "embed-v4.0",
        api_base_url: str = "https://api.cohere.com",
        timeout: int = 120,
        embedding_type: Optional[EmbeddingTypes] = None,
    ):
        """
        Initialize the CohereDocumentImageEmbedder.

        :param api_key: the Cohere API key.
        :param model: the name of the model to use. Supported Models are:
            `"embed-english-v3.0"`, `"embed-english-light-v3.0"`, `"embed-multilingual-v3.0"`,
            `"embed-multilingual-light-v3.0"`, `"embed-english-v2.0"`, `"embed-english-light-v2.0"`,
            `"embed-multilingual-v2.0"`. This list of all supported models can be found in the
            [model documentation](https://docs.cohere.com/docs/models#representation).
        :param input_type: specifies the type of input you're giving to the model. Supported values are
            "search_document", "search_query", "classification" and "clustering". Not
            required for older versions of the embedding models (meaning anything lower than v3), but is required for
            more recent versions (meaning anything bigger than v2).
        :param api_base_url: the Cohere API Base url.
        :param truncate: truncate embeddings that are too long from start or end, ("NONE"|"START"|"END").
            Passing "START" will discard the start of the input. "END" will discard the end of the input. In both
            cases, input is discarded until the remaining input is exactly the maximum input token length for the model.
            If "NONE" is selected, when the input exceeds the maximum input token length an error will be returned.
        :param use_async_client: flag to select the AsyncClient. It is recommended to use
            AsyncClient for applications with many concurrent calls.
        :param timeout: request timeout in seconds.
        :param batch_size: number of Documents to encode at once.
        :param progress_bar: whether to show a progress bar or not. Can be helpful to disable in production deployments
                             to keep the logs clean.
        :param meta_fields_to_embed: list of meta fields that should be embedded along with the Document text.
        :param embedding_separator: separator used to concatenate the meta fields to the Document text.
        :param embedding_type: the type of embeddings to return. Defaults to float embeddings.
            Note that int8, uint8, binary, and ubinary are only valid for v3 models.
        """

        self.meta_field_for_image_path = meta_field_for_image_path
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.embedding_type = embedding_type or EmbeddingTypes.FLOAT
        self._client = ClientV2(
            api_key=api_key.resolve_value(),
            base_url=self.api_base_url,
            timeout=self.timeout,
            client_name="haystack",
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            meta_field_for_image_path=self.meta_field_for_image_path,
            api_key=self.api_key.to_dict(),
            model=self.model,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            embedding_type=self.embedding_type.value,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CohereDocumentImageEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
                Deserialized component.
        """
        init_params = data.get("init_parameters", {})
        deserialize_secrets_inplace(init_params, ["api_key"])

        # Convert embedding_type string to EmbeddingTypes enum value
        init_params["embedding_type"] = EmbeddingTypes.from_str(init_params["embedding_type"])

        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embed a list of `Documents`.

        :param documents: documents to embed.
        :returns:  A dictionary with the following keys:
            - `documents`: documents with the `embedding` field set.
        :raises TypeError: if the input is not a list of `Documents`.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "CohereDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the CohereTextEmbedder."
            )
            raise TypeError(msg)

        if not documents:
            # return early if we were passed an empty list
            return {"documents": []}

        images_to_embed = []
        for doc in documents:
            image_path = doc.meta.get(self.meta_field_for_image_path)
            if not image_path:
                raise ValueError(f"Image path not found in document metadata for document {doc.id}")
            image_content = ImageContent.from_file_path(file_path=image_path)
            images_to_embed.append(f"data:{image_content.mime_type};base64,{image_content.base64_image}")

        all_embeddings = []
        for img in images_to_embed:
            response = self._client.embed(
                texts=None,
                images=[img],  # Cohere allows only one image per request
                model=self.model,
                input_type="image",
                truncate="NONE",
                embedding_types=[self.embedding_type.value],
            )
            ## response.embeddings always returns 5 tuples, one tuple per embedding type
            ## let's take first non None tuple as that's the one we want
            for emb_tuple in response.embeddings:
                # emb_tuple[0] is a str denoting the embedding type (e.g. "float", "int8", etc.)
                if emb_tuple[1] is not None:
                    # ok we have embeddings for this type, let's take all the embeddings (a list of embeddings)
                    all_embeddings.extend(emb_tuple[1])

        for doc, embedding in zip(documents, all_embeddings):
            doc.embedding = embedding
            doc.meta["embedding_source"] = f"meta.{self.meta_field_for_image_path}"

        return {"documents": documents}
