# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack_experimental.components.tools.openapi._openapi import OpenAPIServiceClient, ClientConfiguration
from test.components.tools.openapi.conftest import FastAPITestClient, create_openapi_spec


class TestEdgeCases:

    def test_missing_operation_id(self, test_files_path):
        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_edge_cases.yml"),
                                     request_sender=FastAPITestClient(None))
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
