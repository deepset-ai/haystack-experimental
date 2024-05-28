# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack_experimental.util.openapi import OpenAPISpecification, cohere_converter


class TestOpenAPISchemaConversion:

    def test_serperdev(self, test_files_path):
        spec = OpenAPISpecification.from_file(test_files_path / "yaml" / "serper.yml")
        functions = cohere_converter(schema=spec)

        assert functions
        assert len(functions) == 1
        function = functions[0]
        assert function["name"] == "serperdev_search"
        assert function["description"] == "Search the web with Google"
        assert function["parameter_definitions"] == {"q": {"description": "", "type": "str", "required": True}}

    def test_github(self, test_files_path):
        spec = OpenAPISpecification.from_file(test_files_path / "yaml" / "github_compare.yml")
        functions = cohere_converter(schema=spec)
        assert functions
        assert len(functions) == 1
        function = functions[0]
        assert function["name"] == "compare_branches"
        assert function["description"] == "Compares two branches against one another."
        assert function["parameter_definitions"] == {
            "basehead": {
                "description": "The base branch and head branch to compare."
                " This parameter expects the format `BASE...HEAD`",
                "type": "str",
                "required": True,
            },
            "owner": {
                "description": "The repository owner, usually a company or orgnization",
                "type": "str",
                "required": True,
            },
            "repo": {"description": "The repository itself, the project", "type": "str", "required": True},
        }

    def test_complex_types(self, test_files_path):
        spec = OpenAPISpecification.from_file(test_files_path / "json" / "complex_types_openapi_service.json")
        functions = cohere_converter(schema=spec)

        assert functions
        assert len(functions) == 1
        function = functions[0]
        assert function["name"] == "processPayment"
        assert function["description"] == "Process a new payment using the specified payment method"
        assert function["parameter_definitions"] == {
            "transaction_amount": {"type": "float", "description": "The amount to be paid", "required": True},
            "description": {"type": "str", "description": "A brief description of the payment", "required": True},
            "payment_method_id": {"type": "str", "description": "The payment method to be used", "required": True},
            "payer": {
                "type": "object",
                "description": "Information about the payer, including their name, email, and identification number",
                "properties": {
                    "name": {"type": "str", "description": "The payer's name", "required": True},
                    "email": {"type": "str", "description": "The payer's email address", "required": True},
                    "identification": {
                        "type": "object",
                        "description": "The payer's identification number",
                        "properties": {
                            "type": {
                                "type": "str",
                                "description": "The type of identification document (e.g., CPF, CNPJ)",
                                "required": True,
                            },
                            "number": {"type": "str", "description": "The identification number", "required": True},
                        },
                        "required": True,
                    },
                },
                "required": True,
            },
        }
