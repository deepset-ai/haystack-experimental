[![PyPI - Version](https://img.shields.io/pypi/v/haystack-experimental.svg)](https://pypi.org/project/haystack-experimental)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/haystack-experimental.svg)](https://pypi.org/project/haystack-experimental)
[![Tests](https://github.com/deepset-ai/haystack-experimental/actions/workflows/tests.yml/badge.svg)](https://github.com/deepset-ai/haystack-experimental/actions/workflows/tests.yml)
[![Project release on PyPi](https://github.com/deepset-ai/haystack-experimental/actions/workflows/pypi_release.yml/badge.svg)](https://github.com/deepset-ai/haystack-experimental/actions/workflows/pypi_release.yml)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

# Haystack experimental package

The `haystack-experimental` package provides Haystack users with access to experimental features without immediately
committing to their official release. The main goal is to gather user feedback and iterate on new features quickly.

## Installation

For simplicity, every release of `haystack-experimental` will ship all the available experiments at that time. To
install the latest experimental features, run:

```sh
$ pip install -U haystack-experimental
```

Install from the `main` branch to try the newest features:
```sh
pip install git+https://github.com/deepset-ai/haystack-experimental.git@main
```

> [!IMPORTANT]
> The latest version of the experimental package is only tested against the latest version of Haystack. Compatibility
> with older versions of Haystack is not guaranteed.

## Experiments lifecycle

Each experimental feature has a default lifespan of 3 months starting from the date of the first non-pre-release build
that includes it. Once it reaches the end of its lifespan, the experiment will be either:

- Merged into Haystack core and published in the next minor release, or
- Released as a Core Integration, or
- Dropped.

## Experiments catalog

### Active experiments

| Name                                                                                                                                                                                    | Type                                  | Expected End Date | Dependencies | Cookbook                                                                                                                                                                                                                                                   | Discussion    |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|-------------------|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| [`EmbeddingBasedDocumentSplitter`][8]                                                                                                                                                   | EmbeddingBasedDocumentSplitter        | August 2025       | None         | None                                                                                                                                                                                                                                                       | [Discuss][7]  |
| [`OpenAIChatGenerator`][9]                                                                                                                                                              | Chat Generator Component              | November 2025     | None         | <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/hallucination_score_calculator.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>        | [Discuss][10] |
| [`MarkdownHeaderLevelInferrer`][15]                                                                                                                                                     | Preprocessor                          | January 2025      | None         | None                                                                                                                                                                                                                                                       | [Discuss][16] |
| [`Agent`][17]; [Confirmation Policies][18]; [ConfirmationUIs][19]; [ConfirmationStrategies][20]; [`ConfirmationUIResult` and `ToolExecutionDecision`][21] [HITLBreakpointException][22] | Human in the Loop                     | December 2025     | rich         | None                                                                                                                                                                                                                                                       | [Discuss][23] |
| [`LLMSummarizer`][24]                                                                                                                                                                   | Document Summarizer                   | January 2025      | None         | None                                                                                                                                                                                                                                                       | [Discuss][25] |
| [`InMemoryChatMessageStore`][1]; [`ChatMessageRetriever`][2]; [`ChatMessageWriter`][3]                                                                                                  | Chat Message Store, Retriever, Writer | February 2025     | None         | <a href="https://colab.research.google.com/github/deepset-ai/haystack-cookbook/blob/main/notebooks/conversational_rag_using_memory.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>       | [Discuss][4]  |

[1]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/chat_message_stores/in_memory.py
[2]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/retrievers/chat_message_retriever.py
[3]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/writers/chat_message_writer.py
[4]: https://github.com/deepset-ai/haystack-experimental/discussions/75
[7]: https://github.com/deepset-ai/haystack-experimental/discussions/356
[8]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/preprocessors/embedding_based_document_splitter.py
[9]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/generators/chat/openai.py
[10]: https://github.com/deepset-ai/haystack-experimental/discussions/361
[15]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/preprocessors/md_header_level_inferrer.py  
[16]: https://github.com/deepset-ai/haystack-experimental/discussions/376
[17]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/agents/agent.py
[18]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/agents/human_in_the_loop/policies.py
[19]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/agents/human_in_the_loop/user_interfaces.py
[20]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/agents/human_in_the_loop/strategies.py
[21]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/agents/human_in_the_loop/dataclasses.py
[22]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/agents/human_in_the_loop/errors.py
[23]: https://github.com/deepset-ai/haystack-experimental/discussions/381
[24]: https://github.com/deepset-ai/haystack-experimental/blob/main/haystack_experimental/components/sumarizers/llm_summarizer.py
[25]: https://github.com/deepset-ai/haystack-experimental/discussions/382

### Adopted experiments
| Name                                                                                                                                                                                                                                       | Type                                     | Final release |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|---------------|
| `ChatMessage` refactoring; `Tool` class; tool support in ChatGenerators; `ToolInvoker`                                                                                                                                                     | Tool Calling support                     | 0.4.0         |
| `AsyncPipeline`; `Pipeline` bug fixes and refactoring                                                                                                                                                                                      | AsyncPipeline execution                  | 0.7.0         |
| `LLMMetadataExtractor`                                                                                                                                                                                                                     | Metadata extraction with LLM             | 0.7.0         |
| `Auto-Merging Retriever` & `HierarchicalDocumentSplitter`                                                                                                                                                                                  | Document Splitting & Retrieval Technique | 0.8.0         |
| `Agent`                                                                                                                                                                                                                                    | Simplify Agent development               | 0.8.0         |
| `SuperComponent`                                                                                                                                                                                                                           | Simplify Pipeline development            | 0.8.0         |
| `Pipeline`                                                                                                                                                                                                                                 | Pipeline breakpoints for debugging       | 0.12.0        |
| `ImageContent`; Image Converters; multimodal support in `OpenAIChatGenerator` and `AmazonBedrockChatGenerator`; `ChatPromptBuilder` refactoring; `SentenceTransformersDocumentImageEmbedder`; `LLMDocumentContentExtractor`; new `Routers` | Multimodality                            | 0.12.0        |
| `QueryExpander`                                                                                                                                                                                                                            | Query Expansion Component                | 0.14.3       |
| `MultiQueryEmbeddingRetriever`                                                                                                                                                                                                             | MultiQueryEmbeddingRetriever             | 0.14.3       |
| `MultiQueryTextRetriever`                                                                                                                                                                                                                  | MultiQueryTextRetriever                  | 0.14.3       |


### Discontinued experiments

| Name                   | Type                       | Final release | Cookbook                                                                                                                                 |
|------------------------|----------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `OpenAIFunctionCaller` | Function Calling Component | 0.3.0         | None                                                                                                                                     |
| `OpenAPITool`          | OpenAPITool component      | 0.3.0         | [Notebook](https://github.com/deepset-ai/haystack-experimental/blob/fe20b69b31243f8a3976e4661d9aa8c88a2847d2/examples/openapitool.ipynb) |
| `EvaluationHarness`    | Evaluation orchestrator    | 0.7.0         | None                                                                                                                                     |

## Usage

Experimental new features can be imported like any other Haystack integration package:

```python
from haystack.dataclasses import ChatMessage
from haystack_experimental.components.generators import FoobarGenerator

c = FoobarGenerator()
c.run([ChatMessage.from_user("What's an experiment? Be brief.")])
```

Experiments can also override existing Haystack features. For example, users can opt into an experimental type of
`Pipeline` by just changing the usual import:

```python
# from haystack import Pipeline
from haystack_experimental import Pipeline

pipe = Pipeline()
# ...
pipe.run(...)
```

Some experimental features come with example notebooks that can be found in the [Haystack Cookbook](https://haystack.deepset.ai/cookbook).

## Documentation

Documentation for `haystack-experimental` can be found [here](https://docs.haystack.deepset.ai/reference/experimental-data-classes-api).

## Implementation

Experiments should replicate the namespace of the core package. For example, a new generator:

```python
# in haystack_experimental/components/generators/foobar.py

from haystack import component


@component
class FoobarGenerator:
    ...

```

When the experiment overrides an existing feature, the new symbol should be created at the same path in the experimental
package. This new symbol will override the original in `haystack-ai`: for classes, with a subclass and for bare
functions, with a wrapper. For example:

```python
# in haystack_experiment/src/haystack_experiment/core/pipeline/pipeline.py

from haystack.core.pipeline import Pipeline as HaystackPipeline


class Pipeline(HaystackPipeline):
    # Any new experimental method that doesn't exist in the original class
    def run_async(self, inputs) -> Dict[str, Dict[str, Any]]:
        ...

    # Existing methods with breaking changes to their signature, like adding a new mandatory param
    def to_dict(self, new_param: str) -> Dict[str, Any]:
        # do something with the new parameter
        print(new_param)
        # call the original method
        return super().to_dict()

```

## Contributing

Direct contributions to `haystack-experimental` are not expected, but Haystack maintainers might ask contributors to move pull requests that target the [core repository](https://github.com/deepset-ai/haystack) to this repository.

## Telemetry

As with the Haystack core package, we rely on anonymous usage statistics to determine the impact and usefulness of the experimental features. For more information on what we collect and how we use the data, as well as instructions to opt-out, please refer to our [documentation](https://docs.haystack.deepset.ai/docs/telemetry).
