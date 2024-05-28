# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack_experimental.util.openapi import ClientConfigurationBuilder, OpenAPIServiceClient
from test.util.conftest import FastAPITestClient


class TestEdgeCases:
    def test_invalid_openapi_spec(self):
        builder = ClientConfigurationBuilder()
        with pytest.raises(ValueError, match="Invalid OpenAPI specification"):
            config = builder.with_openapi_spec("invalid_spec.yml").build()
            OpenAPIServiceClient(config)

    def test_missing_operation_id(self, test_files_path):
        builder = ClientConfigurationBuilder()
        config = (
            builder.with_openapi_spec(test_files_path / "yaml" / "openapi_edge_cases.yml")
            .with_http_client(FastAPITestClient(None))
            .build()
        )
        client = OpenAPIServiceClient(config)

        payload = {
            "type": "function",
            "function": {
                "arguments": '{"name": "John", "message": "Hola"}',
                "name": "missingOperationId",
            },
        }
        with pytest.raises(ValueError, match="No operation found with operationId"):
            client.invoke(payload)

    # TODO: Add more tests for edge cases
