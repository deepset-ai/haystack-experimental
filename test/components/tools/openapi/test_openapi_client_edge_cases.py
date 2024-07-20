# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


import pytest

from haystack_experimental.components.tools.openapi._openapi import ClientConfiguration, OpenAPIServiceClient
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

    def test_missing_operation_id_in_operation(self, test_files_path):
        """
        Test that the tool definition is generated correctly when the operationId is missing in the specification.
        """
        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_edge_cases.yml"),
                                     request_sender=FastAPITestClient(None))

        tools = config.get_tools_definitions(),
        tool_def = tools[0][0]
        assert tool_def["type"] == "function"
        assert tool_def["function"]["name"] == "missing_operation_id_get"

    def test_servers_order(self, test_files_path):
        """
        Test that servers defined in different locations in the specification are used correctly.
        """

        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_edge_cases.yml"),
                                     request_sender=FastAPITestClient(None))

        op = config.openapi_spec.find_operation_by_id("servers-order-path")
        assert op.get_server() == "https://inpath.example.com"
        op = config.openapi_spec.find_operation_by_id("servers-order-operation")
        assert op.get_server() == "https://inoperation.example.com"
        op = config.openapi_spec.find_operation_by_id("missing_operation_id_get")
        assert op.get_server() == "http://localhost"
