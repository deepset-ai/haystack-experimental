# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class MarkdownHeaderLevelInferrer:
    """
    Infers and rewrites header levels in markdown text to normalize hierarchy.
    """

    def __init__(self):
        """
        Initializes the MarkdownHeaderLevelInferrer.

        Uses a hardcoded regex pattern to match markdown headers from level 1 to 6.
        """
        self._header_pattern = r"(?m)^(#{1,6}) (.+)$"

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

        processed_docs = []
        for doc in documents:
            if doc.content is None:
                logger.warning(
                    "Document content is None; skipping header level inference.",
                )
                processed_docs.append(doc)
                continue

            matches = list(re.finditer(self._header_pattern, doc.content))
            if not matches:
                logger.info(
                    "No headers found in document{doc_ref}; skipping header level inference.",
                    doc_ref=f" (id: {doc.id})" if hasattr(doc, "id") and doc.id else "",
                )
                processed_docs.append(doc)
                continue

            modified_text = doc.content
            offset = 0
            current_level = 1
            header_stack = [1]

            for i, match in enumerate(matches):
                original_header = match.group(0)
                header_text = match.group(2).strip()

                has_content = False
                if i > 0:
                    prev_end = matches[i - 1].end()
                    current_start = match.start()
                    content_between = doc.content[prev_end:current_start]
                    if content_between is not None:
                        has_content = bool(content_between.strip())

                if i == 0:
                    inferred_level = 1
                elif has_content:
                    inferred_level = current_level
                else:
                    inferred_level = min(current_level + 1, 6)

                current_level = inferred_level
                header_stack = header_stack[:inferred_level]
                while len(header_stack) < inferred_level:
                    header_stack.append(1)

                new_prefix = "#" * inferred_level
                new_header = f"{new_prefix} {header_text}"
                start_pos = match.start() + offset
                end_pos = match.end() + offset
                modified_text = modified_text[:start_pos] + new_header + modified_text[end_pos:]
                offset += len(new_header) - len(original_header)

            logger.info(
                "Rewrote {num_headers} headers with inferred levels in document{doc_ref}.",
                num_headers=len(matches),
                doc_ref=f" (id: {doc.id})" if hasattr(doc, "id") and doc.id else "",
            )
            # Create a new Document with updated content, preserving other fields
            processed_docs.append(
                Document(
                    id=doc.id if hasattr(doc, "id") and doc.id is not None else "",
                    content=modified_text,
                    blob=getattr(doc, "blob", None),
                    meta=getattr(doc, "meta", {}) if getattr(doc, "meta", None) is not None else {},
                    score=getattr(doc, "score", None),
                    embedding=getattr(doc, "embedding", None),
                )
            )

        return {"documents": processed_docs}
