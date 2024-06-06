# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LLMProvider(Enum):
    """
    Enum for the supported LLM providers.
    """
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
