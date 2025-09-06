# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat import AzureOpenAIChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_callable, deserialize_secrets_inplace
from tqdm import tqdm

with LazyImport(message="Run 'pip install tiktoken'") as tiktoken_import:
    import tiktoken

with LazyImport(message="Run 'pip install \"amazon-bedrock-haystack>=1.0.2\"'") as amazon_bedrock_generator:
    from haystack_integrations.components.generators.amazon_bedrock import AmazonBedrockChatGenerator

with LazyImport(message="Run 'pip install \"google-vertex-haystack>=2.0.0\"'") as vertex_ai_gemini_generator:
    from haystack_integrations.components.generators.google_vertex.chat.gemini import VertexAIGeminiChatGenerator
    from vertexai.generative_models import GenerationConfig

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    OPENAI_AZURE = "openai_azure"
    AWS_BEDROCK = "aws_bedrock"
    GOOGLE_VERTEX = "google_vertex"

    @staticmethod
    def from_str(string: str) -> "LLMProvider":
        """
        Convert a string to a LLMProvider enum.
        """
        provider_map = {e.value: e for e in LLMProvider}
        provider = provider_map.get(string)
        if provider is None:
            msg = f"Invalid LLMProvider '{string}'Supported LLMProviders are: {list(provider_map.keys())}"
            raise ValueError(msg)
        return provider


@component
class Summarizer:
    """
    Summarizes text using a language model.

    It's inspired by code from the OpenAI blog post: https://cookbook.openai.com/examples/summarizing_long_documents

    To run this example you need to install the `wikipedia` package:

    ```bash
        pip install wikipedia
    ```

    Example
    ```python
    import wikipedia
    from haystack_experimental.components.summarizer.summarizer import Summarizer
    from haystack import Document

    page = wikipedia.page("Berlin")
    textual_content = page.content
    doc = Document(content=textual_content)

    summarizer = Summarizer(generator_api="openai")
    summarizer.run(documents=[doc])
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        generator_api: Union[str, LLMProvider],
        generator_api_params: Optional[dict[str, Any]] = None,
        system_prompt: Optional[str] = "Rewrite this text in summarized form.",
        summary_detail: float = 0,
        minimum_chunk_size: Optional[int] = 500,
        chunk_delimiter: str = ".",
        summarize_recursively=False,
    ):
        """
        Initialize the Summarizer component.

        :param generator_api: The API to use for summarization.
        :param generator_api_params: Parameters for the generator API.
        :param system_prompt: The prompt to instruct the LLM to summarise text, if not given defaults to:
            "Rewrite this text in summarized form."
        :param summary_detail: The level of detail for the summary (0-1), defaults to 0.
        :param minimum_chunk_size: The minimum token count per chunk, defaults to 500
        :param chunk_delimiter: The character used to split the text into chunks, defaults to "."
        :param summarize_recursively: Whether to use previous summaries as context, defaults to False.
        """
        self.detail = summary_detail
        self.minimum_chunk_size = minimum_chunk_size
        self.chunk_delimiter = chunk_delimiter
        self.system_prompt = system_prompt
        self.summarize_recursively = summarize_recursively
        self.generator_api = (
            generator_api if isinstance(generator_api, LLMProvider) else LLMProvider.from_str(generator_api)
        )
        self.generator_api_params = generator_api_params or {}
        self.llm_provider = self._init_generator(self.generator_api, self.generator_api_params)

    @staticmethod
    def _init_generator(
        generator_api: LLMProvider, generator_api_params: Optional[dict[str, Any]]
    ) -> Union[
        OpenAIChatGenerator, AzureOpenAIChatGenerator, "AmazonBedrockChatGenerator", "VertexAIGeminiChatGenerator"
    ]:
        """
        Initialize the chat generator based on the specified API provider and parameters.
        """
        if generator_api == LLMProvider.OPENAI:
            return OpenAIChatGenerator(**generator_api_params)
        elif generator_api == LLMProvider.OPENAI_AZURE:
            return AzureOpenAIChatGenerator(**generator_api_params)
        elif generator_api == LLMProvider.AWS_BEDROCK:
            amazon_bedrock_generator.check()
            return AmazonBedrockChatGenerator(**generator_api_params)
        elif generator_api == LLMProvider.GOOGLE_VERTEX:
            vertex_ai_gemini_generator.check()
            return VertexAIGeminiChatGenerator(**generator_api_params)
        else:
            raise ValueError(f"Unsupported generator API: {generator_api}")

    def warm_up(self):
        """
        Warm up the LLM provider component.
        """
        if hasattr(self.llm_provider, "warm_up"):
            self.llm_provider.warm_up()

    def to_dict(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        llm_provider = self.llm_provider.to_dict()

        return default_to_dict(
            self,
            llm_provider=llm_provider,
            summary_detail=self.detail,
            minimum_chunk_size=self.minimum_chunk_size,
            chunk_delimiter=self.chunk_delimiter,
            summarize_recursively=self.summarize_recursively,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Summarizer":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        init_params = data.get("init_parameters", {})

        if "generator_api" in init_params:
            data["init_parameters"]["generator_api"] = LLMProvider.from_str(data["init_parameters"]["generator_api"])

        if "generator_api_params" in init_params:
            # Azure
            azure_openai_keys = ["azure_ad_token"]

            # AWS
            aws_bedrock_keys = [
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_region_name",
                "aws_profile_name",
            ]
            deserialize_secrets_inplace(
                data["init_parameters"]["generator_api_params"],
                keys=["api_key"] + azure_openai_keys + aws_bedrock_keys,
            )

            # VertexAI
            if "generation_config" in init_params["generator_api_params"]:
                data["init_parameters"]["generation_config"] = GenerationConfig.from_dict(
                    init_params["generator_api_params"]["generation_config"]
                )

            # common
            serialized_callback_handler = init_params["generator_api_params"].get("streaming_callback")
            if serialized_callback_handler:
                data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)

    def num_tokens(self, text: str) -> int:
        """
        Estimates the token count for a given text.

        If we are using OpenAI based models (e.g., GPT-4), this method uses the tiktoken library to estimate the
        token count.

        If we are using other models, for now, we use a simple heuristic of 4 characters.
        "You can think of tokens as pieces of words that are roughly 4 characters of typical English text"
        source: https://techcommunity.microsoft.com/blog/machinelearningblog/introducing-llama-2-on-azure/3881233

        But we should consider using the exact tokenizer from each model to get the exact token count.

        :param text: The text to tokenize
        :returns:
            The estimated token count
        """
        if self.generator_api == LLMProvider.OPENAI or self.generator_api == LLMProvider.OPENAI_AZURE:
            tiktoken_import.check()
            model_name = self.generator_api_params.get("model", "gpt-4-turbo")
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))

        # fallback to approximate tokenization - assuming most LLMs average ~4 characters per token
        return len(text) // 4

    def _chunk_on_delimiter(self, text: str, max_tokens: int, delimiter: str) -> list[str]:
        """
        Chunks a text into smaller pieces based on a maximum token count and a delimiter.

        :param text: The text to be chunked
        :param max_tokens: The maximum token count for each chunk
        :param delimiter: The delimiter used to split the text into chunks

        :returns:
            List of text chunks with delimiters reattached
        """

        chunks = text.split(delimiter)

        # combine chunks to optimize token usage
        combined_chunks, _, dropped_chunk_count = self._combine_chunks(
            chunks, max_tokens, chunk_delimiter=delimiter, add_ellipsis_overflow=True
        )

        # warning if chunks were dropped
        if dropped_chunk_count > 0:
            msg = f"{dropped_chunk_count} chunks were dropped due to overflow. "
            logger.warning(msg)

        # add delimiters
        return [f"{chunk}{delimiter}" for chunk in combined_chunks]

    def _combine_chunks(
        self, chunks: list[str], max_tokens: int, chunk_delimiter="\n\n", add_ellipsis_overflow=False
    ) -> tuple[list[str], list[list[int]], int]:
        """
        Combines chunks into larger blocks without exceeding a specified token count.

        :param chunks: The list of chunks to be combined
        :param max_tokens: Maximum token count for each combined chunk
        :param chunk_delimiter: String used to join chunks together
        :param add_ellipsis_overflow: Whether to add "..." for chunks that exceed max_tokens

        :returns:
            Tuple containing:
                - List of combined text blocks
                - List of original chunk indices for each combined block
                - Count of chunks dropped due to overflow
        """
        dropped_chunk_count = 0
        output = []  # Combined chunks
        output_indices = []  # Original indices of chunks in each combined block
        candidate: list[str] = []  # Current chunk being built
        candidate_indices: list[int] = []

        for chunk_i, chunk in enumerate(chunks):
            # check if single chunk exceeds max_tokens
            if self.num_tokens(chunk) > max_tokens:
                logger.warning("Chunk overflow detected")
                if add_ellipsis_overflow and self.num_tokens(chunk_delimiter.join(candidate + ["..."])) <= max_tokens:
                    candidate.append("...")
                    dropped_chunk_count += 1
                continue  # Skip this chunk as it would break downstream assumptions

            # check if adding this chunk would exceed max_tokens
            candidate_with_new_chunk = candidate + [chunk]
            extended_candidate_text = chunk_delimiter.join(candidate_with_new_chunk)

            if self.num_tokens(extended_candidate_text) > max_tokens:
                # save current candidate and start a new one
                output.append(chunk_delimiter.join(candidate))
                output_indices.append(candidate_indices)
                candidate = [chunk]
                candidate_indices = [chunk_i]
            else:
                # add chunk to current candidate
                candidate.append(chunk)
                candidate_indices.append(chunk_i)

        # if it's not empty add the final candidate
        if len(candidate) > 0:
            output.append(chunk_delimiter.join(candidate))
            output_indices.append(candidate_indices)

        return output, output_indices, dropped_chunk_count

    def _prepare_text_chunks(self, text, detail, minimum_chunk_size, chunk_delimiter):
        """Prepares text chunks based on detail level."""
        # calculate optimal chunk count
        max_chunks = len(self._chunk_on_delimiter(text, minimum_chunk_size, chunk_delimiter))
        min_chunks = 1
        num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))

        # determine appropriate chunk size
        document_length = self.num_tokens(text)
        chunk_size = max(minimum_chunk_size, document_length // num_chunks)

        return self._chunk_on_delimiter(text, chunk_size, chunk_delimiter)

    def _process_chunks(self, text_chunks, summarize_recursively):
        """
        Processes each chunk individually, asking the LLM to summarize it, and accumulates all the summaries.

        The parameter `summarize_recursively` allows to use previous summaries as context for the next chunk.
        """
        accumulated_summaries = []

        for chunk in tqdm(text_chunks):
            if summarize_recursively and accumulated_summaries:
                accumulated_summaries_string = "\n\n".join(accumulated_summaries)
                user_message_content = (
                    f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
                )
            else:
                user_message_content = chunk

            # prepare the message and make the LLM call
            messages = [ChatMessage.from_system(self.system_prompt), ChatMessage.from_user(user_message_content)]
            # ToDo: some error handling here
            result = self.llm_provider.run(messages=messages)
            accumulated_summaries.append(result["replies"][0].text)

        return accumulated_summaries

    def summarize(
        self,
        text: str,
        detail: float,
        minimum_chunk_size: int,
        summarize_recursively=False,
    ) -> str:
        """
        Summarizes text by splitting it into optimally-sized chunks and processing each with an LLM.

        :param text: Text to summarize
        :param detail: Detail level (0-1) where 0 is most concise and 1 is most detailed
        :param minimum_chunk_size: Minimum token count per chunk
        :param summarize_recursively: Whether to use previous summaries as context

        :returns:
            The textual content summarized by the LLM.
        """

        if not 0 <= detail <= 1:
            raise ValueError("Detail must be between 0 and 1")

        # calculate "optimal" chunking parameters
        text_chunks = self._prepare_text_chunks(text, detail, minimum_chunk_size, self.chunk_delimiter)

        # process chunks and accumulate summaries
        accumulated_summaries = self._process_chunks(text_chunks, summarize_recursively)

        # combine all summaries
        return "\n\n".join(accumulated_summaries)

    @component.output_types(summary=list[Document])
    def run(
        self,
        documents: list[Document],
        detail: Optional[float] = None,
        minimum_chunk_size: Optional[int] = None,
        summarize_recursively: Optional[bool] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Run the summarizer on a list of documents.

        :param documents: List of documents to summarize
        :param detail: The level of detail for the summary (0-1), defaults to 0 overwriting the component's default.
        :param minimum_chunk_size: The minimum token count per chunk, defaults to 500 overwriting the
                                   component's default.
        :param system_prompt: If given it will overwrite prompt given at init time or the default one.
        :param summarize_recursively: Whether to use previous summaries as context, defaults to False overwriting the
                                      component's default.
        """

        # let's allow to change some of the parameters at run time
        detail = self.detail if detail is None else detail
        minimum_chunk_size = self.minimum_chunk_size if minimum_chunk_size is None else minimum_chunk_size
        summarize_recursively = self.summarize_recursively if summarize_recursively is None else summarize_recursively
        self.system_prompt = system_prompt if system_prompt else self.system_prompt

        for doc in documents:
            summary = self.summarize(
                doc.content,
                detail=detail,
                minimum_chunk_size=minimum_chunk_size,  # type: ignore # already checked, cannot be None here
                summarize_recursively=summarize_recursively,
            )
            doc.meta["summary_detail"] = summary

        return {"documents": documents}
