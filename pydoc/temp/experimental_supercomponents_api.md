---
title: "SuperComponents"
id: experimental-supercomponents-api
description: "Pipelines wrapped as components."
slug: "/experimental-supercomponents-api"
---

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer"></a>

## Module haystack\_experimental.super\_components.indexers.sentence\_transformers\_document\_indexer

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer"></a>

### SentenceTransformersDocumentIndexer

A document indexer that takes a list of documents, embeds them using SentenceTransformers, and stores them.

Usage:

```python
>>> from haystack import Document
>>> from haystack.document_stores.in_memory import InMemoryDocumentStore
>>> document_store = InMemoryDocumentStore()
>>> doc = Document(content="I love pizza!")
>>> indexer = SentenceTransformersDocumentIndexer(document_store=document_store)
>>> indexer.warm_up()
>>> result = indexer.run(documents=[doc])
>>> print(result)
{'documents_written': 1}
>>> document_store.count_documents()
1
```

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.__init__"></a>

#### SentenceTransformersDocumentIndexer.\_\_init\_\_

```python
def __init__(
        document_store: DocumentStore,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(
            ["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        trust_remote_code: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        precision: Literal["float32", "int8", "uint8", "binary",
                           "ubinary"] = "float32",
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE) -> None
```

Initialize the SentenceTransformersDocumentIndexer component.

**Arguments**:

- `document_store`: The document store where the documents should be stored.
- `model`: The embedding model to use (local path or Hugging Face model ID).
- `device`: The device to use for loading the model.
- `token`: The API token to download private models from Hugging Face.
- `prefix`: String to add at the beginning of each document text.
- `suffix`: String to add at the end of each document text.
- `batch_size`: Number of documents to embed at once.
- `progress_bar`: If True, shows a progress bar when embedding documents.
- `normalize_embeddings`: If True, embeddings are L2 normalized.
- `meta_fields_to_embed`: List of metadata fields to embed along with the document text.
- `embedding_separator`: Separator used to concatenate metadata fields to document text.
- `trust_remote_code`: If True, allows custom models and scripts.
- `truncate_dim`: Dimension to truncate sentence embeddings to.
- `model_kwargs`: Additional keyword arguments for model initialization.
- `tokenizer_kwargs`: Additional keyword arguments for tokenizer initialization.
- `config_kwargs`: Additional keyword arguments for model configuration.
- `precision`: The precision to use for the embeddings.
- `duplicate_policy`: The duplicate policy to use when writing documents.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.to_dict"></a>

#### SentenceTransformersDocumentIndexer.to\_dict

```python
def to_dict() -> Dict[str, Any]
```

Serialize this instance to a dictionary.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.from_dict"></a>

#### SentenceTransformersDocumentIndexer.from\_dict

```python
@classmethod
def from_dict(cls, data: Dict[str,
                              Any]) -> "SentenceTransformersDocumentIndexer"
```

Load an instance of this component from a dictionary.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.show"></a>

#### SentenceTransformersDocumentIndexer.show

```python
def show(server_url: str = "https://mermaid.ink",
         params: Optional[dict] = None,
         timeout: int = 30) -> None
```

Display an image representing this SuperComponent's underlying pipeline in a Jupyter notebook.

This function generates a diagram of the Pipeline using a Mermaid server and displays it directly in
the notebook.

**Arguments**:

- `server_url`: The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
info on how to set up your own Mermaid server.
- `params`: Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
Supported keys:
- format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
- type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
- theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
- bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
- width: Width of the output image (integer).
- height: Height of the output image (integer).
- scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
- fit: Whether to fit the diagram size to the page (PDF only, boolean).
- paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
- landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.
- `timeout`: Timeout in seconds for the request to the Mermaid server.

**Raises**:

- `PipelineDrawingError`: If the function is called outside of a Jupyter notebook or if there is an issue with rendering.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.draw"></a>

#### SentenceTransformersDocumentIndexer.draw

```python
def draw(path: Path,
         server_url: str = "https://mermaid.ink",
         params: Optional[dict] = None,
         timeout: int = 30) -> None
```

Save an image representing this SuperComponent's underlying pipeline to the specified file path.

This function generates a diagram of the Pipeline using the Mermaid server and saves it to the provided path.

**Arguments**:

- `path`: The file path where the generated image will be saved.
- `server_url`: The base URL of the Mermaid server used for rendering (default: 'https://mermaid.ink').
See https://github.com/jihchi/mermaid.ink and https://github.com/mermaid-js/mermaid-live-editor for more
info on how to set up your own Mermaid server.
- `params`: Dictionary of customization parameters to modify the output. Refer to Mermaid documentation for more details
Supported keys:
- format: Output format ('img', 'svg', or 'pdf'). Default: 'img'.
- type: Image type for /img endpoint ('jpeg', 'png', 'webp'). Default: 'png'.
- theme: Mermaid theme ('default', 'neutral', 'dark', 'forest'). Default: 'neutral'.
- bgColor: Background color in hexadecimal (e.g., 'FFFFFF') or named format (e.g., '!white').
- width: Width of the output image (integer).
- height: Height of the output image (integer).
- scale: Scaling factor (1–3). Only applicable if 'width' or 'height' is specified.
- fit: Whether to fit the diagram size to the page (PDF only, boolean).
- paper: Paper size for PDFs (e.g., 'a4', 'a3'). Ignored if 'fit' is true.
- landscape: Landscape orientation for PDFs (boolean). Ignored if 'fit' is true.
- `timeout`: Timeout in seconds for the request to the Mermaid server.

**Raises**:

- `PipelineDrawingError`: If there is an issue with rendering or saving the image.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.warm_up"></a>

#### SentenceTransformersDocumentIndexer.warm\_up

```python
def warm_up() -> None
```

Warms up the SuperComponent by warming up the wrapped pipeline.

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.run"></a>

#### SentenceTransformersDocumentIndexer.run

```python
def run(**kwargs: Any) -> dict[str, Any]
```

Runs the wrapped pipeline with the provided inputs.

Steps:
1. Maps the inputs from kwargs to pipeline component inputs
2. Runs the pipeline
3. Maps the pipeline outputs to the SuperComponent's outputs

**Arguments**:

- `kwargs`: Keyword arguments matching the SuperComponent's input names

**Returns**:

Dictionary containing the SuperComponent's output values

<a id="haystack_experimental.super_components.indexers.sentence_transformers_document_indexer.SentenceTransformersDocumentIndexer.run_async"></a>

#### SentenceTransformersDocumentIndexer.run\_async

```python
async def run_async(**kwargs: Any) -> dict[str, Any]
```

Runs the wrapped pipeline with the provided inputs async.

Steps:
1. Maps the inputs from kwargs to pipeline component inputs
2. Runs the pipeline async
3. Maps the pipeline outputs to the SuperComponent's outputs

**Arguments**:

- `kwargs`: Keyword arguments matching the SuperComponent's input names

**Raises**:

- `TypeError`: If the pipeline is not an AsyncPipeline

**Returns**:

Dictionary containing the SuperComponent's output values
