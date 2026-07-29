[![PyPI - Version](https://img.shields.io/pypi/v/haystack-experimental.svg)](https://pypi.org/project/haystack-experimental)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-experimental.svg)](https://pypi.org/project/haystack-experimental)

# Haystack experimental package

> [!WARNING]
> **This project is archived and no longer maintained.**
>
> `0.19.0.post1` is the **final release**. There will be no further releases, bug fixes, or compatibility updates,
> and the repository is read-only.
>
> - **Most experiments graduated into Haystack itself.** If you are looking for `Agent`, `Tool` and tool calling,
>   `AsyncPipeline`, pipeline breakpoints, multimodality, `SuperComponent`, `QueryExpander`, Human-in-the-Loop and
>   more, they all ship in [`haystack-ai`](https://pypi.org/project/haystack-ai) now — see
>   [Graduated experiments](#graduated-experiments) below for the full list and use the
>   [Haystack documentation](https://docs.haystack.deepset.ai/docs/intro).
> - **The remaining experiments were discontinued** rather than graduated — see
>   [Discontinued experiments](#discontinued-experiments). They are not part of Haystack and will not be maintained
>   anywhere.
> - **If you depend on a discontinued experiment**, pin the final release explicitly. It will remain installable from
>   PyPI, but it is only tested against the version of Haystack that was current in February 2026 and will drift out
>   of compatibility with newer `haystack-ai` releases:
>
>   ```sh
>   pip install "haystack-experimental==0.19.0.post1"
>   ```

The `haystack-experimental` package gave Haystack users early access to experimental features without immediately
committing to their official release, so that we could gather feedback and iterate quickly. Each experiment had a
limited lifespan, after which it was either merged into Haystack core, released as a Core Integration, or dropped.

That process has now concluded. This README is kept as the record of where each experiment ended up.

## Experiments catalog

### Graduated experiments

These experiments were adopted into Haystack core and are available in `haystack-ai`. The version column is the last
release of `haystack-experimental` that contained the experimental copy.

| Name                                                                                                                                                                                                                                      | Type                                     | Final release |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|---------------|
| `ChatMessage` refactoring; `Tool` class; tool support in ChatGenerators; `ToolInvoker`                                                                                                                                                     | Tool Calling support                     | 0.4.0         |
| `AsyncPipeline`; `Pipeline` bug fixes and refactoring                                                                                                                                                                                     | AsyncPipeline execution                  | 0.7.0         |
| `LLMMetadataExtractor`                                                                                                                                                                                                                    | Metadata extraction with LLM             | 0.7.0         |
| `Auto-Merging Retriever` & `HierarchicalDocumentSplitter`                                                                                                                                                                                 | Document Splitting & Retrieval Technique | 0.8.0         |
| `Agent`                                                                                                                                                                                                                                   | Simplify Agent development               | 0.8.0         |
| `SuperComponent`                                                                                                                                                                                                                          | Simplify Pipeline development            | 0.8.0         |
| `Pipeline`                                                                                                                                                                                                                                | Pipeline breakpoints for debugging       | 0.12.0        |
| `ImageContent`; Image Converters; multimodal support in `OpenAIChatGenerator` and `AmazonBedrockChatGenerator`; `ChatPromptBuilder` refactoring; `SentenceTransformersDocumentImageEmbedder`; `LLMDocumentContentExtractor`; new `Routers` | Multimodality                            | 0.12.0        |
| `QueryExpander`                                                                                                                                                                                                                           | Query Expansion Component                | 0.14.3        |
| `MultiQueryEmbeddingRetriever`                                                                                                                                                                                                             | MultiQueryEmbeddingRetriever             | 0.14.3        |
| `MultiQueryTextRetriever`                                                                                                                                                                                                                  | MultiQueryTextRetriever                  | 0.14.3        |
| `EmbeddingBasedDocumentSplitter`                                                                                                                                                                                                           | Document Splitting                       | 0.15.2        |
| `Confirmation Policies`; `ConfirmationUIs`; `BlockingConfirmationStrategy`; `ConfirmationUIResult`; `ToolExecutionDecision`                                                                                                                | Human in the Loop                        | 0.16.0        |
| `Mem0MemoryStore`                                                                                                                                                                                                                         | MemoryStore                              | 0.19.0        |

### Discontinued experiments

These experiments were **not** adopted into Haystack. They exist only in the release listed below.

| Name                                                                                | Type                              | Final release | Cookbook                                                                                                                                 | Discussion    |
|-------------------------------------------------------------------------------------|-----------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `OpenAIFunctionCaller`                                                              | Function Calling Component        | 0.3.0         | None                                                                                                                                     | --            |
| `OpenAPITool`                                                                       | OpenAPITool component             | 0.3.0         | [Notebook](https://github.com/deepset-ai/haystack-experimental/blob/fe20b69b31243f8a3976e4661d9aa8c88a2847d2/examples/openapitool.ipynb) | [Discuss][5]  |
| `EvaluationHarness`                                                                 | Evaluation orchestrator           | 0.7.0         | None                                                                                                                                     | [Discuss][6]  |
| `Agent`; `BreakpointConfirmationStrategy`; `HITLBreakpointException`                 | Human in the Loop via Breakpoints | 0.19.0        | None                                                                                                                                     | [Discuss][23] |
| [`InMemoryChatMessageStore`][1]; [`ChatMessageRetriever`][2]; [`ChatMessageWriter`][3] | Chat Message Store, Retriever, Writer | 0.19.0    | <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/conversational_rag_using_memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [Discuss][4]  |
| [`OpenAIChatGenerator`][9] (hallucination risk scoring)                             | Chat Generator Component          | 0.19.0        | <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/hallucination_score_calculator.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | [Discuss][10] |
| [`MarkdownHeaderLevelInferrer`][15]                                                 | Preprocessor                      | 0.19.0        | None                                                                                                                                     | [Discuss][16] |
| [`LLMSummarizer`][24]                                                               | Document Summarizer               | 0.19.0        | None                                                                                                                                     | [Discuss][25] |

[1]: https://github.com/deepset-ai/haystack-experimental/blob/v0.19.0/haystack_experimental/chat_message_stores/in_memory.py
[2]: https://github.com/deepset-ai/haystack-experimental/blob/v0.19.0/haystack_experimental/components/retrievers/chat_message_retriever.py
[3]: https://github.com/deepset-ai/haystack-experimental/blob/v0.19.0/haystack_experimental/components/writers/chat_message_writer.py
[4]: https://github.com/deepset-ai/haystack-experimental/discussions/75
[5]: https://github.com/deepset-ai/haystack-experimental/discussions/79
[6]: https://github.com/deepset-ai/haystack-experimental/discussions/74
[9]: https://github.com/deepset-ai/haystack-experimental/blob/v0.19.0/haystack_experimental/components/generators/chat/openai.py
[10]: https://github.com/deepset-ai/haystack-experimental/discussions/361
[15]: https://github.com/deepset-ai/haystack-experimental/blob/v0.19.0/haystack_experimental/components/preprocessors/md_header_level_inferrer.py
[16]: https://github.com/deepset-ai/haystack-experimental/discussions/376
[23]: https://github.com/deepset-ai/haystack-experimental/discussions/381
[24]: https://github.com/deepset-ai/haystack-experimental/blob/v0.19.0/haystack_experimental/components/summarizers/llm_summarizer.py
[25]: https://github.com/deepset-ai/haystack-experimental/discussions/382

## Usage

Experimental features were imported like any other Haystack integration package:

```python
from haystack import Document
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_experimental.components.summarizers import LLMSummarizer

summarizer = LLMSummarizer(chat_generator=OpenAIChatGenerator())
summarizer.run(documents=[Document(content="...")])
```

Some experiments came with example notebooks in the [Haystack Cookbook](https://haystack.deepset.ai/cookbook).

## Telemetry

As with the Haystack core package, this package collected anonymous usage statistics. For more information on what was
collected and how to opt out, refer to the [telemetry documentation](https://docs.haystack.deepset.ai/docs/telemetry).
