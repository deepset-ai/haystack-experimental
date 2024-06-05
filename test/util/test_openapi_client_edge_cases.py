# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack_experimental.util.openapi import OpenAPIServiceClient, ClientConfiguration
from test.util.conftest import FastAPITestClient


class TestEdgeCases:

    def test_missing_operation_id(self, test_files_path):
        config = ClientConfiguration(openapi_spec=test_files_path / "yaml" / "openapi_edge_cases.yml",
                                     http_client=FastAPITestClient(None))
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
