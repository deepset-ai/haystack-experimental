# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class MarkdownHeaderLevelInferrer:
    """
    Infers and rewrites header levels in Markdown text to normalize hierarchy.
    """

    def __init__(self):
        """
        Initializes the MarkdownHeaderLevelInferrer.
        """
        self._header_pattern = re.compile(r"(?m)^(#{1,6}) (.+)$")

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict:
        """
        Infers and rewrites the header levels in the content for documents that use uniform header levels.

        Args:
            documents (list[Document]): List of Document objects to process.

        Returns:
            dict: A dictionary with the key 'documents' containing the processed Document objects.
        """
        logger.debug("Inferring and rewriting header levels for documents")
        processed_docs = [self._process_document(doc) for doc in documents]
        return {"documents": processed_docs}

    def _process_document(self, doc: Document) -> Document:
        """Processes a single document, inferring and rewriting header levels."""
        if doc.content is None:
            logger.warning(f"Document {getattr(doc, 'id', '')} content is None; skipping header level inference.")
            return doc

        matches = list(re.finditer(self._header_pattern, doc.content))
        if not matches:
            logger.info(f"No headers found in document{self._doc_ref(doc)}; skipping header level inference.")
            return doc

        modified_text = self._rewrite_headers(doc.content, matches)
        logger.info(f"Rewrote {len(matches)} headers with inferred levels in document{self._doc_ref(doc)}.")
        return self._build_final_document(doc, modified_text)

    def _rewrite_headers(self, content: str, matches: list[re.Match]) -> str:
        """Rewrites the headers in the content with inferred levels."""
        modified_text = content
        offset = 0
        current_level = 1

        for i, match in enumerate(matches):
            original_header = match.group(0)
            header_text = match.group(2).strip()

            has_content = self._has_content_between_headers(content, matches, i)
            inferred_level = self._infer_level(i, current_level, has_content)
            current_level = inferred_level

            new_header = f"{'#' * inferred_level} {header_text}"
            start_pos = match.start() + offset
            end_pos = match.end() + offset
            modified_text = modified_text[:start_pos] + new_header + modified_text[end_pos:]
            offset += len(new_header) - len(original_header)

        return modified_text

    def _has_content_between_headers(self, content: str, matches: list[re.Match], i: int) -> bool:
        """Checks if there is content between the previous and current header."""
        if i == 0:
            return False
        prev_end = matches[i - 1].end()
        current_start = matches[i].start()
        content_between = content[prev_end:current_start]
        return bool(content_between.strip())

    def _infer_level(self, i: int, current_level: int, has_content: bool) -> int:
        """Infers the header level for the current header."""
        if i == 0:
            return 1
        if has_content:
            return current_level
        return min(current_level + 1, 6)

    def _build_final_document(self, doc: Document, new_content: str) -> Document:
        """Creates a new Document with updated content, preserving other fields."""
        return Document(
            id=getattr(doc, "id", "") or "",
            content=new_content,
            blob=getattr(doc, "blob", None),
            meta=getattr(doc, "meta", {}) or {},
            score=getattr(doc, "score", None),
            embedding=getattr(doc, "embedding", None),
        )

    def _doc_ref(self, doc: Document) -> str:
        """Returns a string reference for the document id."""
        doc_id = getattr(doc, "id", None)
        return f" (id: {doc_id})" if doc_id else ""
