# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_experimental.components.generators.chat.openai import (  # noqa: I001 (otherwise we end up with partial imports)
    OpenAIChatGenerator,
)


__all__ = [
    "OpenAIChatGenerator",
]
