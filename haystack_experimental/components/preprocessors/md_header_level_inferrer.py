# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Optional

from haystack import component, logging


@component
class MarkdownHeaderLevelInferrer:
    """
    Infers and rewrites header levels in markdown text to normalize hierarchy.
    """

    def __init__(self, header_pattern: str = r"(?m)^(#{1,6}) (.+)$"):
        self._header_pattern = header_pattern

    @component.output_types(content=str)
    def run(self, content: str, doc_id: Optional[str] = None) -> dict[str, str]:
        logger = logging.getLogger(__name__)
        logger.debug("Inferring and rewriting header levels")

        matches = list(re.finditer(self._header_pattern, content))
        if not matches:
            logger.info(
                "No headers found in document{doc_ref}; skipping header level inference.",
                doc_ref=f" (id: {doc_id})" if doc_id else "",
            )
            return {"content": content}

        modified_text = content
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
                content_between = content[prev_end:current_start].strip()
                has_content = bool(content_between)

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

        logger.info("Rewrote {num_headers} headers with inferred levels.", num_headers=len(matches))
        return {"content": modified_text}

