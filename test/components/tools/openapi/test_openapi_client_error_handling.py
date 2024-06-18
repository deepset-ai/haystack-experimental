# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import json

import pytest
from fastapi import FastAPI, HTTPException

from haystack_experimental.components.tools.openapi._openapi import OpenAPIServiceClient, HttpClientError, \
    ClientConfiguration
from test.components.tools.openapi.conftest import FastAPITestClient, create_openapi_spec


def create_error_handling_app() -> FastAPI:
    app = FastAPI()

    @app.get("/error/{status_code}")
    def raise_http_error(status_code: int):
        raise HTTPException(status_code=status_code, detail=f"HTTP {status_code} error")

    return app


class TestErrorHandling:
    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 500])
    def test_http_error_handling(self, test_files_path, status_code):
        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_error_handling.yml"),
                                     request_sender=FastAPITestClient(create_error_handling_app()))
        client = OpenAPIServiceClient(config)
        json_error = {"status_code": status_code}
        payload = {
            "type": "function",
            "function": {
                "arguments": json.dumps(json_error),
                "name": "raiseHttpError",
            },
        }
        with pytest.raises(HttpClientError) as exc_info:
            client.invoke(payload)

        assert str(status_code) in str(exc_info.value)
