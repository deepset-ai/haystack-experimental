# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat import AzureOpenAIChatGenerator, OpenAIChatGenerator
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage
from haystack.lazy_imports import LazyImport
from haystack.utils import deserialize_chatgenerator_inplace
from tqdm import tqdm

with LazyImport(message="Run 'pip install tiktoken'") as tiktoken_import:
    import tiktoken

logger = logging.getLogger(__name__)


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
    from haystack_experimental.components.summarizers.summarizer import Summarizer
    from haystack import Document
    from haystack.components.generators.chat import OpenAIChatGenerator

    page = wikipedia.page("Berlin")
    textual_content = page.content
    doc = Document(content=textual_content)

    chat_generator = OpenAIChatGenerator(model="gpt-4")
    summarizer = Summarizer(chat_generator=chat_generator)
    summarizer.run(documents=[doc])
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        chat_generator: ChatGenerator,
        system_prompt: Optional[str] = "Rewrite this text in summarized form.",
        summary_detail: float = 0,
        minimum_chunk_size: Optional[int] = 500,
        chunk_delimiter: str = ".",
        summarize_recursively: bool = False,
    ):
        """
        Initialize the Summarizer component.

        :param chat_generator: A ChatGenerator instance to use for summarization.
        :param system_prompt: The prompt to instruct the LLM to summarise text, if not given defaults to:
            "Rewrite this text in summarized form."
        :param summary_detail: The level of detail for the summary (0-1), defaults to 0.
        :param minimum_chunk_size: The minimum token count per chunk, defaults to 500
        :param chunk_delimiter: The character used to split the text into chunks, defaults to "."
        :param summarize_recursively: Whether to use previous summaries as context, defaults to False.
        """
        self._chat_generator = chat_generator
        self.detail = summary_detail
        self.minimum_chunk_size = minimum_chunk_size
        self.chunk_delimiter = chunk_delimiter
        self.system_prompt = system_prompt
        self.summarize_recursively = summarize_recursively

    def warm_up(self):
        """
        Warm up the chat generator component.
        """
        if hasattr(self._chat_generator, "warm_up"):
            self._chat_generator.warm_up()

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self._chat_generator, name="chat_generator"),
            system_prompt=self.system_prompt,
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
        deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")
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
        if isinstance(self._chat_generator, (OpenAIChatGenerator, AzureOpenAIChatGenerator)):
            tiktoken_import.check()
            model_name = getattr(self._chat_generator, "model", "gpt-4-turbo")
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
        self, chunks: list[str], max_tokens: int, chunk_delimiter: str = "\n\n", add_ellipsis_overflow: bool = False
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
        accumulated_summaries: list[str] = []

        for chunk in tqdm(text_chunks):
            if summarize_recursively and accumulated_summaries:
                accumulated_summaries_string = "\n\n".join(accumulated_summaries)
                user_message_content = (
                    f"Previous summaries:\n\n{accumulated_summaries_string}\n\nText to summarize next:\n\n{chunk}"
                )
            else:
                user_message_content = chunk

            # prepare the message and make the LLM call
            # self.system_prompt is not None
            messages = [ChatMessage.from_system(self.system_prompt), ChatMessage.from_user(user_message_content)]  # type: ignore
            # ToDo: some error handling here
            result = self._chat_generator.run(messages=messages)
            accumulated_summaries.append(result["replies"][0].text)

        return accumulated_summaries

    def summarize(
        self,
        text: str,
        detail: float,
        minimum_chunk_size: int,
        summarize_recursively: bool = False,
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
        *,
        documents: list[Document],
        detail: Optional[float] = None,
        minimum_chunk_size: Optional[int] = None,
        summarize_recursively: Optional[bool] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, list[Document]]:
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
            if doc.content is None or doc.content == "":
                logger.warning("Document ID {doc_id} has an empty content. Skipping this document.", doc_id=doc.id)
                continue
            summary = self.summarize(
                doc.content,
                detail=detail,
                minimum_chunk_size=minimum_chunk_size,  # type: ignore # already checked, cannot be None here
                summarize_recursively=summarize_recursively,
            )
            doc.meta["summary_detail"] = summary

        return {"documents": documents}
