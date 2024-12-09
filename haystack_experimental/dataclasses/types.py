# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, TypedDict


class OpenAPIKwargs(TypedDict, total=False):
    """
    TypedDict for OpenAPI configuration kwargs.

    Contains all supported configuration options for Tool.from_openapi_spec()
    """

    credentials: Any  # API credentials (e.g., API key, auth token)
    request_sender: Callable[[Dict[str, Any]], Dict[str, Any]]  # Custom HTTPrequest sender function
    allowed_operations: List[str]  # A list of operations to include in the OpenAPI client.
