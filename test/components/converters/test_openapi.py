# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import json
import sys
import tempfile

import pytest

from haystack_experimental.components.converters import OpenAPIServiceToFunctions
from haystack.dataclasses import ByteStream


@pytest.fixture
def json_serperdev_openapi_spec():
    serper_spec = """
            {
                "openapi": "3.0.0",
                "info": {
                    "title": "SerperDev",
                    "version": "1.0.0",
                    "description": "API for performing search queries"
                },
                "servers": [
                    {
                        "url": "https://google.serper.dev"
                    }
                ],
                "paths": {
                    "/search": {
                        "post": {
                            "operationId": "search",
                            "description": "Search the web with Google",
                            "requestBody": {
                                "required": true,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "q": {
                                                    "type": "string"
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "responses": {
                                "200": {
                                    "description": "Successful response",
                                    "content": {
                                        "application/json": {
                                            "schema": {
                                                "type": "object",
                                                "properties": {
                                                    "searchParameters": {
                                                        "type": "undefined"
                                                    },
                                                    "knowledgeGraph": {
                                                        "type": "undefined"
                                                    },
                                                    "answerBox": {
                                                        "type": "undefined"
                                                    },
                                                    "organic": {
                                                        "type": "undefined"
                                                    },
                                                    "topStories": {
                                                        "type": "undefined"
                                                    },
                                                    "peopleAlsoAsk": {
                                                        "type": "undefined"
                                                    },
                                                    "relatedSearches": {
                                                        "type": "undefined"
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "security": [
                                {
                                    "apikey": []
                                }
                            ]
                        }
                    }
                },
                "components": {
                    "securitySchemes": {
                        "apikey": {
                            "type": "apiKey",
                            "name": "x-api-key",
                            "in": "header"
                        }
                    }
                }
            }
            """
    return serper_spec


@pytest.fixture
def yaml_serperdev_openapi_spec():
    serper_spec = """
            openapi: 3.0.0
            info:
              title: SerperDev
              version: 1.0.0
              description: API for performing search queries
            servers:
              - url: 'https://google.serper.dev'
            paths:
              /search:
                post:
                  operationId: search
                  description: Search the web with Google
                  requestBody:
                    required: true
                    content:
                      application/json:
                        schema:
                          type: object
                          properties:
                            q:
                              type: string
                  responses:
                    '200':
                      description: Successful response
                      content:
                        application/json:
                          schema:
                            type: object
                            properties:
                              searchParameters:
                                type: undefined
                              knowledgeGraph:
                                type: undefined
                              answerBox:
                                type: undefined
                              organic:
                                type: undefined
                              topStories:
                                type: undefined
                              peopleAlsoAsk:
                                type: undefined
                              relatedSearches:
                                type: undefined
                  security:
                    - apikey: []
            components:
              securitySchemes:
                apikey:
                  type: apiKey
                  name: x-api-key
                  in: header
            """
    return serper_spec


@pytest.fixture
def fn_definition_transform():
    return lambda function_def: {"type": "function", "function": function_def}


class TestOpenAPIServiceToFunctions:
    # test we can extract functions from openapi spec given
    def test_run_with_bytestream_source(self, json_serperdev_openapi_spec, fn_definition_transform):
        service = OpenAPIServiceToFunctions()
        spec_stream = ByteStream.from_string(json_serperdev_openapi_spec)
        result = service.run(sources=[spec_stream])
        assert len(result["functions"]) == 1
        fc = result["functions"][0]

        # check that fc definition is as expected
        assert fc == fn_definition_transform(
            {
                "name": "search",
                "description": "Search the web with Google",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        )

    @pytest.mark.skipif(
        sys.platform in ["win32", "cygwin"],
        reason="Can't run on Windows Github CI, need access temp file but windows does not allow it",
    )
    def test_run_with_file_source(self, json_serperdev_openapi_spec, fn_definition_transform):
        # test we can extract functions from openapi spec given in file
        service = OpenAPIServiceToFunctions()
        # write the spec to NamedTemporaryFile and check that it is parsed correctly
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(json_serperdev_openapi_spec.encode("utf-8"))
            tmp.seek(0)
            result = service.run(sources=[tmp.name])
            assert len(result["functions"]) == 1
            fc = result["functions"][0]

            # check that fc definition is as expected
            assert fc == fn_definition_transform(
                {
                    "name": "search",
                    "description": "Search the web with Google",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                }
            )

    def test_run_with_invalid_bytestream_source(self, caplog):
        # test invalid source
        service = OpenAPIServiceToFunctions()
        with pytest.raises(ValueError, match="Invalid OpenAPI specification"):
            service.run(sources=[ByteStream.from_string("")])

    def test_complex_types_conversion(self, test_files_path, fn_definition_transform):
        # ensure that complex types from OpenAPI spec are converted to the expected format in OpenAI function calling
        service = OpenAPIServiceToFunctions()
        result = service.run(sources=[test_files_path / "json" / "complex_types_openapi_service.json"])
        assert len(result["functions"]) == 1

        with open(test_files_path / "json" / "complex_types_openai_spec.json") as openai_spec_file:
            desired_output = json.load(openai_spec_file)
        assert result["functions"][0] == fn_definition_transform(desired_output)

    def test_simple_and_complex_at_once(self, test_files_path, json_serperdev_openapi_spec, fn_definition_transform):
        # ensure multiple functions are extracted from multiple paths in OpenAPI spec
        service = OpenAPIServiceToFunctions()
        sources = [
            ByteStream.from_string(json_serperdev_openapi_spec),
            test_files_path / "json" / "complex_types_openapi_service.json",
        ]
        result = service.run(sources=sources)
        assert len(result["functions"]) == 2

        with open(test_files_path / "json" / "complex_types_openai_spec.json") as openai_spec_file:
            desired_output = json.load(openai_spec_file)
        assert result["functions"][0] == fn_definition_transform(
            {
                "name": "search",
                "description": "Search the web with Google",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
        )
        assert result["functions"][1] == fn_definition_transform(desired_output)
