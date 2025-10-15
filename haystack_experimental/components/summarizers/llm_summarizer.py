# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.preprocessors import RecursiveDocumentSplitter
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage
from haystack.utils import deserialize_chatgenerator_inplace
from tqdm import tqdm

logger = logging.getLogger(__name__)


@component
class LLMSummarizer:
    """
    Summarizes text using a language model.

    It's inspired by code from the OpenAI blog post: https://cookbook.openai.com/examples/summarizing_long_documents

    Example
    ```python
    from haystack_experimental.components.summarizers.summarizer import Summarizer
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack import Document

    text = ("Machine learning is a subset of artificial intelligence that provides systems "
            "the ability to automatically learn and improve from experience without being "
            "explicitly programmed. The process of learning begins with observations or data. "
            "Supervised learning algorithms build a mathematical model of sample data, known as "
            "training data, in order to make predictions or decisions. Unsupervised learning "
            "algorithms take a set of data that contains only inputs and find structure in the data. "
            "Reinforcement learning is an area of machine learning where an agent learns to behave "
            "in an environment by performing actions and seeing the results. Deep learning uses "
            "artificial neural networks to model complex patterns in data. Neural networks consist "
            "of layers of connected nodes, each performing a simple computation.")

    doc = Document(content=text)
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
        split_overlap: int = 0,
    ):
        """
        Initialize the Summarizer component.

        :param chat_generator: A ChatGenerator instance to use for summarization.
        :param system_prompt: The prompt to instruct the LLM to summarise text, if not given defaults to:
            "Rewrite this text in summarized form."
        :param summary_detail: The level of detail for the summary (0-1), defaults to 0.
            This parameter controls the trade-off between conciseness and completeness by adjusting how many
            chunks the text is divided into. At detail=0, the text is processed as a single chunk (or very few
            chunks), producing the most concise summary. At detail=1, the text is split into the maximum number
            of chunks allowed by minimum_chunk_size, enabling more granular analysis and detailed summaries.
            The formula uses linear interpolation: num_chunks = 1 + detail * (max_chunks - 1), where max_chunks
            is determined by dividing the document length by minimum_chunk_size.
        :param minimum_chunk_size: The minimum token count per chunk, defaults to 500
        :param chunk_delimiter: The character used to determine separator priority.
            "." uses sentence-based splitting, "\n" uses paragraph-based splitting, defaults to "."
        :param summarize_recursively: Whether to use previous summaries as context, defaults to False.
        :param split_overlap: Number of tokens to overlap between consecutive chunks, defaults to 0.
        """
        self._chat_generator = chat_generator
        self.summary_detail = summary_detail
        self.minimum_chunk_size = minimum_chunk_size
        self.chunk_delimiter = chunk_delimiter
        self.system_prompt = system_prompt
        self.summarize_recursively = summarize_recursively
        self.split_overlap = split_overlap

        # Map chunk_delimiter to an appropriate separator strategy
        separators = LLMSummarizer._get_separators_from_delimiter(chunk_delimiter)

        # Initialize RecursiveDocumentSplitter
        # Note: split_length will be updated dynamically based on detail parameter
        self._document_splitter = RecursiveDocumentSplitter(
            split_length=minimum_chunk_size if minimum_chunk_size else 500,
            split_overlap=split_overlap,
            split_unit="token",
            separators=separators,
        )

    @staticmethod
    def _get_separators_from_delimiter(delimiter: str) -> list[str]:
        """
        Map the delimiter to an appropriate list of separators for RecursiveDocumentSplitter.

        :param delimiter: The delimiter character
        :returns: List of separators in order of preference
        """
        if delimiter == ".":
            # Sentence-focused splitting
            return ["\n\n", "sentence", "\n", " "]
        elif delimiter == "\n":
            # Paragraph-focused splitting
            return ["\n\n", "\n", "sentence", " "]
        else:
            # Custom delimiter - prioritize it
            return ["\n\n", delimiter, "\n", " "]

    def warm_up(self):
        """
        Warm up the chat generator and document splitter components.
        """
        # Warm up chat generator
        if hasattr(self._chat_generator, "warm_up"):
            self._chat_generator.warm_up()

        # Warm up document splitter (needed for sentence splitting and tokenization)
        if hasattr(self._document_splitter, "warm_up"):
            self._document_splitter.warm_up()

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
            summary_detail=self.summary_detail,
            minimum_chunk_size=self.minimum_chunk_size,
            chunk_delimiter=self.chunk_delimiter,
            summarize_recursively=self.summarize_recursively,
            split_overlap=self.split_overlap,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMSummarizer":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        init_params = data.get("init_parameters", {})

        # Deserialize chat_generator
        deserialize_chatgenerator_inplace(init_params, key="chat_generator")

        return default_from_dict(cls, data)

    def num_tokens(self, text: str) -> int:
        """
        Estimates the token count for a given text.

        Uses the RecursiveDocumentSplitter's tokenization logic for consistency.

        :param text: The text to tokenize
        :returns:
            The estimated token count
        """
        # Use the document splitter's tokenization method for consistency
        return self._document_splitter._chunk_length(text)

    def _prepare_text_chunks(self, text, detail, minimum_chunk_size, chunk_delimiter):
        """
        Prepares text chunks based on detail level using RecursiveDocumentSplitter.

        The detail parameter (0-1) controls the granularity through linear interpolation:
        - detail=0: Creates fewer, larger chunks (most concise summary)
        - detail=1: Creates more, smaller chunks (most detailed summary)

        The formula calculates: num_chunks = 1 + detail * (max_chunks - 1), where max_chunks is the
        document_length divided by minimum_chunk_size. This interpolates between processing the entire
        text as one chunk (detail=0) and splitting it into the maximum number of chunks that respect
        the minimum_chunk_size constraint (detail=1). Higher detail values allow the LLM to analyze
        smaller portions of text more carefully, preserving more information at the cost of longer
        processing time and potentially longer summaries.

        :param text: The text to chunk
        :param detail: Detail level (0-1)
        :param minimum_chunk_size: Minimum token count per chunk
        :param chunk_delimiter: Delimiter for separator selection
        :returns: List of text chunks
        """
        document_length = self.num_tokens(text)
        max_chunks = max(1, document_length // minimum_chunk_size)
        min_chunks = 1

        num_chunks = int(min_chunks + detail * (max_chunks - min_chunks))
        num_chunks = max(1, num_chunks)  # Ensure at least 1 chunk

        chunk_size = max(minimum_chunk_size, document_length // num_chunks)
        self._document_splitter.split_length = chunk_size
        if chunk_delimiter != self.chunk_delimiter:
            self._document_splitter.separators = self._get_separators_from_delimiter(chunk_delimiter)

        temp_doc = Document(content=text)
        result = self._document_splitter.run(documents=[temp_doc])
        text_chunks = [doc.content for doc in result["documents"]]

        return text_chunks

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
            # self.system_prompt is not None where due to the default value in the constructor
            messages = [ChatMessage.from_system(self.system_prompt), ChatMessage.from_user(user_message_content)]  # type: ignore
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

        :raises ValueError: If detail is not between 0 and 1
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

        :raises RuntimeError: If the component wasn't warmed up.
        """

        if not self._document_splitter._is_warmed_up:
            raise RuntimeError("The Summarizer component wasn't warmed up. Call 'warm_up()' before calling 'run()'.")

        # let's allow to change some of the parameters at run time
        detail = self.summary_detail if detail is None else detail
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
            doc.meta["summary"] = summary

        return {"documents": documents}
