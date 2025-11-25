# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document
from haystack_experimental.components.preprocessors import MarkdownHeaderLevelInferrer


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


def test_empty_documents_list():
    inferrer = MarkdownHeaderLevelInferrer()
    result = inferrer.run([])
    assert result["documents"] == []


def test_document_with_none_content():
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=None)
    result = inferrer.run([doc])
    assert result["documents"][0].content is None


def test_document_with_empty_content():
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content="")
    result = inferrer.run([doc])
    assert result["documents"][0].content == ""


def test_headers_with_trailing_spaces():
    text = "## Header 1   \nContent\n## Header 2   \nMore content"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    expected = "# Header 1\nContent\n# Header 2\nMore content"
    assert content == expected


def test_headers_with_leading_spaces():
    text = "  ## Header 1\nContent\n  ## Header 2\nMore content"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    # Headers with leading spaces should not match the pattern
    assert result["documents"][0].content == text


def test_maximum_header_level():
    text = "## H1\n## H2\n## H3\n## H4\n## H5\n## H6\n## H7\n## H8"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    expected = "# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6\n###### H7\n###### H8"
    assert content == expected


def test_multiple_documents():
    text1 = "## Title 1\nContent 1"
    text2 = "## Title 2\nContent 2"
    inferrer = MarkdownHeaderLevelInferrer()
    docs = [Document(content=text1), Document(content=text2)]
    result = inferrer.run(docs)

    assert len(result["documents"]) == 2
    assert result["documents"][0].content == "# Title 1\nContent 1"
    assert result["documents"][1].content == "# Title 2\nContent 2"


def test_headers_with_special_characters():
    text = "## Header with émojis 🚀\nContent\n## Header with numbers 123\nMore content"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    expected = "# Header with émojis 🚀\nContent\n# Header with numbers 123\nMore content"
    assert result["documents"][0].content == expected


def test_headers_with_markdown_formatting():
    text = "## Header with **bold** text\nContent\n## Header with *italic* text\nMore content"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    expected = "# Header with **bold** text\nContent\n# Header with *italic* text\nMore content"
    assert result["documents"][0].content == expected


def test_very_long_content():
    lines = ["## Header " + str(i) + "\nContent for header " + str(i) for i in range(50)]
    text = "\n".join(lines)
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])

    # verify first header becomes level 1, others follow the pattern
    content = result["documents"][0].content
    assert content.startswith("# Header 0")
    assert "# Header 1" in content
    assert len(content.split("\n")) == len(text.split("\n"))
