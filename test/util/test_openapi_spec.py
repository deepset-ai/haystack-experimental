# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from haystack_experimental.util.openapi import OpenAPISpecification


class TestOpenAPISpecification:

    #  can be initialized from a dictionary
    def test_initialized_from_dictionary(self):
        spec_dict = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {"summary": "Get all users", "responses": {"200": {"description": "Successful response"}}}
                }
            },
        }
        openapi_spec = OpenAPISpecification.from_dict(spec_dict)
        assert openapi_spec.spec_dict == spec_dict

    #  can be initialized from a string
    def test_initialized_from_string(self):
        content = """
        openapi: 3.0.0
        info:
          title: Test API
          version: 1.0.0
        servers:
            - url: https://api.example.com
        paths:
          /users:
            get:
              summary: Get all users
              responses:
                '200':
                  description: Successful response
        """
        openapi_spec = OpenAPISpecification.from_str(content)
        assert openapi_spec.spec_dict == {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {"summary": "Get all users", "responses": {"200": {"description": "Successful response"}}}
                }
            },
        }

    #  can be initialized from a file
    def test_initialized_from_file(self, tmp_path):
        content = """
        openapi: 3.0.0
        info:
          title: Test API
          version: 1.0.0
        servers:
            - url: https://api.example.com
        paths:
          /users:
            get:
              summary: Get all users
              responses:
                '200':
                  description: Successful response
        """
        file_path = tmp_path / "spec.yaml"
        file_path.write_text(content)
        openapi_spec = OpenAPISpecification.from_file(file_path)
        assert openapi_spec.spec_dict == {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {
                "/users": {
                    "get": {"summary": "Get all users", "responses": {"200": {"description": "Successful response"}}}
                }
            },
        }

    #  can get all paths
    def test_get_all_paths(self):
        spec_dict = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "servers": [{"url": "https://api.example.com"}],
            "paths": {"/users": {}, "/products": {}, "/orders": {}},
        }
        openapi_spec = OpenAPISpecification(spec_dict)
        paths = openapi_spec.get_paths()
        assert paths == {"/users": {}, "/products": {}, "/orders": {}}

    #  raises ValueError if initialized from an invalid schema
    def test_raises_value_error_invalid_schema(self):
        spec_dict = {"info": {"title": "Test API", "version": "1.0.0"}, "paths": {"/users": {}}}
        with pytest.raises(ValueError):
            OpenAPISpecification(spec_dict)

    #  Should return the raw OpenAPI specification dictionary with resolved references.
    def test_return_raw_spec_with_resolved_references(self, test_files_path):
        spec = OpenAPISpecification.from_file(test_files_path / "json" / "complex_types_openapi_service.json")
        raw_spec = spec.to_dict(resolve_references=True)

        assert "$ref" not in str(raw_spec)
        assert "#/" not in str(raw_spec)

        # verify that we can serialize the raw spec to a string
        schema_ser = json.dumps(raw_spec, indent=2)

        # and that the serialized string does not contain any $ref or #/ references
        assert "$ref" not in schema_ser
        assert "#/" not in schema_ser

        # and that we can deserialize the serialized string back to a dictionary
        schema = json.loads(schema_ser)
        assert "$ref" not in schema
        assert "#/" not in schema

        assert schema == raw_spec
