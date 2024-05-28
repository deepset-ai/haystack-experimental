# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from haystack_experimental.util.openapi import HttpClient, HttpClientError


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent.parent / "test_files"


class FastAPITestClient(HttpClient):

    def __init__(self, app: FastAPI):
        self.app = app
        self.client = TestClient(app)

    def strip_host(self, url: str) -> str:
        parsed_url = urlparse(url)
        new_path = parsed_url.path
        if parsed_url.query:
            new_path += "?" + parsed_url.query
        return new_path

    def send_request(self, request: dict) -> dict:
        method = request["method"]
        # OAS spec will list a server URL, but FastAPI doesn't need it for local testing, in fact it will fail
        # if the URL has a host. So we strip it here.
        url = self.strip_host(request["url"])
        headers = request.get("headers", {})
        params = request.get("params", {})
        json_data = request.get("json", None)
        auth = request.get("auth", None)
        cookies = request.get("cookies", {})

        try:
            response = self.client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_data,
                auth=auth,
                cookies=cookies,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Handle HTTP errors
            raise HttpClientError(f"HTTP error occurred: {e}") from e
