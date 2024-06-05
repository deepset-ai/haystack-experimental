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

    #  raises ValueError if initialized from an invalid schema
    def test_raises_value_error_invalid_schema(self):
        spec_dict = {"info": {"title": "Test API", "version": "1.0.0"}, "paths": {"/users": {}}}
        with pytest.raises(ValueError):
            OpenAPISpecification(spec_dict)
