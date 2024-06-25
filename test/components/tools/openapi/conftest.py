# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from haystack.utils.url_validation import is_valid_http_url

from haystack_experimental.components.tools.openapi._openapi import HttpClientError
from haystack_experimental.components.tools.openapi.types import OpenAPISpecification


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent.parent.parent.parent / "test_files"


def create_openapi_spec(openapi_spec: Union[Path, str]) -> OpenAPISpecification:
    if isinstance(openapi_spec, (str, Path)) and os.path.isfile(openapi_spec):
        return OpenAPISpecification.from_file(openapi_spec)
    elif isinstance(openapi_spec, str):
        if is_valid_http_url(openapi_spec):
            return OpenAPISpecification.from_url(openapi_spec)
        else:
            return OpenAPISpecification.from_str(openapi_spec)
    else:
        raise ValueError(
            "Invalid OpenAPI specification format. Expected file path or dictionary."
        )


class FastAPITestClient:

    def __init__(self, app: FastAPI):
        self.app = app
        self.client = TestClient(app)

    def strip_host(self, url: str) -> str:
        parsed_url = urlparse(url)
        new_path = parsed_url.path
        if parsed_url.query:
            new_path += "?" + parsed_url.query
        return new_path

    def __call__(self, request: dict) -> dict:
        # OAS spec will list a server URL, but FastAPI doesn't need it for local testing, in fact it will fail
        # if the URL has a host. So we strip it here.
        url = self.strip_host(request["url"])
        try:
            response = self.client.request(
                request["method"],
                url,
                headers=request.get("headers", {}),
                params=request.get("params", {}),
                json=request.get("json", None),
                auth=request.get("auth", None),
                cookies=request.get("cookies", {}),
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Handle HTTP errors
            raise HttpClientError(f"HTTP error occurred: {e}") from e
