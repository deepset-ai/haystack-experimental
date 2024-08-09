# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Dict


def normalize_tool_definition(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes the given tool definition by adjusting its properties to LLM requirements.

    While various LLMs have slightly different requirements for tool definitions, we normalize them to a common
    format that is compatible with OpenAI, Anthropic, and Cohere LLMs:
    - tool names have to match the pattern ^[a-zA-Z0-9_]+$ and are truncated to 64 characters
    - tool/parameter descriptions are truncated to 1024 characters

    For reference on tool definition formats, see:
        - https://cookbook.openai.com/examples/how_to_call_functions_with_chat_models#basic-concepts
        - https://docs.anthropic.com/en/docs/build-with-claude/tool-use
        - https://docs.cohere.com/docs/tool-use

    :param data: The function calling definition(s) to normalize.
    :returns: A normalized function calling definition.
    """
    normalized_data: Dict[str, Any] = {}
    for key, value in data.items():
        # all LLMs tool definitions have tool (function) name and description on the same level
        # if we find it then normalize the function name
        if key == "name" and "description" in data.keys():
            normalized_data[key] = normalize_function_name(value)
        elif key == "description":
            normalized_data[key] = value[:1024]
        elif isinstance(value, dict):
            # recursively normalize nested descriptions (e.g. tool parameters)
            normalized_data[key] = normalize_tool_definition(value)
        else:
            normalized_data[key] = value
    return normalized_data


def normalize_function_name(name: str) -> str:
    """
    Normalizes the function name to match the LLM function naming requirements.

    While various LLMs have slightly different requirements for tool (function) names, we normalize them to
    a common format that is compatible with OpenAI, Anthropic, and Cohere LLMs:
    - The function name must match the pattern ^[a-zA-Z0-9_]+$
    - The function name must be truncated to 64 characters

    :param name: The original function name.
    :returns: A normalized function name that matches the allowed pattern.
    """
    # Replace characters not allowed in the pattern with underscores
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
    # Remove leading and trailing underscores and truncate to 64 characters
    return normalized.strip("_")[:64]
