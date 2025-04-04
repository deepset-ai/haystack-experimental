# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Literal, Optional

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.preprocessors.document_cleaner import DocumentCleaner
from haystack.components.preprocessors.document_splitter import DocumentSplitter, Language
from haystack.utils import deserialize_callable, serialize_callable

from haystack_experimental.core.super_component import SuperComponent


@component
class DocumentPreProcessor(SuperComponent):
    """
    A SuperComponent that cleans documents and then splits them.

    This component composes a DocumentCleaner followed by a DocumentSplitter in a single pipeline.
    It takes a list of documents as input and returns a processed list of documents.

    Usage:
    ```python
    from haystack import Document
    doc = Document(content="I love pizza!")
    preprocessor = DocumentPreProcessor()
    results = preprocessor.run(documents=[doc])
    print(result["documents"])
    ```
    """

    def __init__( # pylint: disable=R0917
        self,
        # --- DocumentCleaner arguments ---
        remove_empty_lines: bool = True,
        remove_extra_whitespaces: bool = True,
        remove_repeated_substrings: bool = False,
        keep_id: bool = False,
        remove_substrings: Optional[List[str]] = None,
        remove_regex: Optional[str] = None,
        unicode_normalization: Optional[Literal["NFC", "NFKC", "NFD", "NFKD"]] = None,
        ascii_only: bool = False,
        # --- DocumentSplitter arguments ---
        split_by: Literal["function", "page", "passage", "period", "word", "line", "sentence"] = "word",
        split_length: int = 250,
        split_overlap: int = 0,
        split_threshold: int = 0,
        splitting_function: Optional[Callable[[str], List[str]]] = None,
        respect_sentence_boundary: bool = False,
        language: Language = "en",
        use_split_rules: bool = True,
        extend_abbreviations: bool = True,
    ) -> None:
        """
        Initialize a DocumentPreProcessor that first cleans documents and then splits them.

        **Cleaner Params**:
        :param remove_empty_lines: If `True`, removes empty lines.
        :param remove_extra_whitespaces: If `True`, removes extra whitespaces.
        :param remove_repeated_substrings: If `True`, remove repeated substrings like headers/footers across pages.
        :param keep_id: If `True`, keeps the original document IDs.
        :param remove_substrings: A list of strings to remove from the document content.
        :param remove_regex: A regex pattern whose matches will be removed from the document content.
        :param unicode_normalization: Unicode normalization form to apply to the text, e.g. `"NFC"`.
        :param ascii_only: If `True`, convert text to ASCII only.

        **Splitter Params**:
        :param split_by: The unit of splitting: "function", "page", "passage", "period", "word", "line", or "sentence".
        :param split_length: The maximum number of units (words, lines, pages, etc.) in each split.
        :param split_overlap: The number of overlapping units between consecutive splits.
        :param split_threshold: The minimum number of units per split. If a split is smaller than this, it's merged
            with the previous split.
        :param splitting_function: A custom function for splitting if `split_by="function"`.
        :param respect_sentence_boundary: If `True`, splits by words but tries not to break inside a sentence.
        :param language: Language used by the sentence tokenizer if `split_by="sentence"` or
            `respect_sentence_boundary=True`.
        :param use_split_rules: Whether to apply additional splitting heuristics for the sentence splitter.
        :param extend_abbreviations: Whether to extend the sentence splitter with curated abbreviations for certain
            languages.
        """
        # Store arguments for serialization
        self.remove_empty_lines = remove_empty_lines
        self.remove_extra_whitespaces = remove_extra_whitespaces
        self.remove_repeated_substrings = remove_repeated_substrings
        self.keep_id = keep_id
        self.remove_substrings = remove_substrings
        self.remove_regex = remove_regex
        self.unicode_normalization = unicode_normalization
        self.ascii_only = ascii_only

        self.split_by = split_by
        self.split_length = split_length
        self.split_overlap = split_overlap
        self.split_threshold = split_threshold
        self.splitting_function = splitting_function
        self.respect_sentence_boundary = respect_sentence_boundary
        self.language = language
        self.use_split_rules = use_split_rules
        self.extend_abbreviations = extend_abbreviations

        # Instantiate sub-components
        cleaner = DocumentCleaner(
            remove_empty_lines=self.remove_empty_lines,
            remove_extra_whitespaces=self.remove_extra_whitespaces,
            remove_repeated_substrings=self.remove_repeated_substrings,
            keep_id=self.keep_id,
            remove_substrings=self.remove_substrings,
            remove_regex=self.remove_regex,
            unicode_normalization=self.unicode_normalization,
            ascii_only=self.ascii_only,
        )

        splitter = DocumentSplitter(
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            splitting_function=self.splitting_function,
            respect_sentence_boundary=self.respect_sentence_boundary,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )

        # Build the Pipeline
        pp = Pipeline()
        pp.add_component("cleaner", cleaner)
        pp.add_component("splitter", splitter)

        # Connect the cleaner output to splitter
        pp.connect("cleaner.documents", "splitter.documents")

        # Define how pipeline inputs/outputs map to sub-component inputs/outputs
        input_mapping = {
            # The pipeline input "documents" feeds into "cleaner.documents"
            "documents": ["cleaner.documents"]
        }
        # The pipeline output "documents" comes from "splitter.documents"
        output_mapping = {"splitter.documents": "documents"}

        # Initialize the SuperComponent
        super(DocumentPreProcessor, self).__init__(
            pipeline=pp,
            input_mapping=input_mapping,
            output_mapping=output_mapping
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this instance to a dictionary.
        """
        data = default_to_dict(
            self,
            remove_empty_lines=self.remove_empty_lines,
            remove_extra_whitespaces=self.remove_extra_whitespaces,
            remove_repeated_substrings=self.remove_repeated_substrings,
            keep_id=self.keep_id,
            remove_substrings=self.remove_substrings,
            remove_regex=self.remove_regex,
            unicode_normalization=self.unicode_normalization,
            ascii_only=self.ascii_only,
            split_by=self.split_by,
            split_length=self.split_length,
            split_overlap=self.split_overlap,
            split_threshold=self.split_threshold,
            respect_sentence_boundary=self.respect_sentence_boundary,
            language=self.language,
            use_split_rules=self.use_split_rules,
            extend_abbreviations=self.extend_abbreviations,
        )

        if self.splitting_function:
            data["init_parameters"]["splitting_function"] = serialize_callable(self.splitting_function)

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentPreProcessor":
        """
        Load this instance from a dictionary.
        """
        if "splitting_function" in data["init_parameters"]:
            data["init_parameters"]["splitting_function"] = deserialize_callable(
                data["init_parameters"]["splitting_function"]
            )

        return default_from_dict(cls, data)
