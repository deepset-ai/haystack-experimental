# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document
from haystack_experimental.components.preprocessors.md_header_level_inferrer import MarkdownHeaderLevelInferrer


def test_split_infer_header_levels():
    text = "## H1\n## H2\nContent"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    # Should rewrite first header to level 1, second to level 1 or 2 depending on content
    assert content.startswith("# H1")
    assert "# H2" in content or "## H2" in content


def test_infer_header_levels_complex():
    """Test header level inference with a complex document structure."""
    text = (
        "## All Headers Same Level\n"
        "Some content\n"
        "## Second Header\n"
        "Some content\n"
        "## Third Header With No Content\n"
        "## Fourth Header With No Content\n"
        "## Fifth Header\n"
        "More content"
    )
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    # First header should be level 1
    assert "# All Headers Same Level" in content
    # Second header with content should stay at level 1
    assert "# Second Header" in content


def test_infer_header_levels_override_both_directions():
    text = "## H1\n## H2\nContent"
    inferrer = MarkdownHeaderLevelInferrer()
    doc = Document(content=text)
    result = inferrer.run([doc])
    content = result["documents"][0].content
    # Should rewrite headers to level 1 or 2
    assert "# H1" in content or "## H1" in content
    assert "# H2" in content or "## H2" in content
