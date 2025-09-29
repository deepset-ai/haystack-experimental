# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document
from haystack_experimental.components.preprocessors.md_header_level_inferrer import MarkdownHeaderLevelInferrer


def test_single_header_level_inference():
    text = "## H1\nSome content\n## H2\nContent"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    # Expect the first header to be rewritten to level 1, second to level 1 (since content follows)
    expected = "# H1\nSome content\n# H2\nContent"
    assert content == expected


def test_header_level_increase_on_consecutive_headers():
    text = "## H1\n## H2\n## H3"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    # Expect the first header to be level 1, the next two to increase in level
    expected = "# H1\n## H2\n### H3"
    assert content == expected


def test_no_headers():
    text = "This is just some text without headers."
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    assert content == text


def test_complex_structure():
    text = (
        "## Title\n"
        "## Section\n"
        "Section content\n"
        "## Subsection\n"
        "Subsection content\n"
        "## Another Section\n"
        "## Another Subsection\n"
        "Even more content\n"
        "## Final Section\n"
        "Final content"
    )
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    expected = (
        "# Title\n"
        "## Section\n"
        "Section content\n"
        "## Subsection\n"
        "Subsection content\n"
        "## Another Section\n"
        "### Another Subsection\n"
        "Even more content\n"
        "### Final Section\n"
        "Final content"
    )
    assert content == expected
