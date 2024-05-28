# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from haystack_experimental.util.openapi import ClientConfigurationBuilder, OpenAPIServiceClient
from test.util.conftest import FastAPITestClient

"""
Tests OpenAPIServiceClient with three FastAPI apps for different parameter types:

- **greet_mix_params_body**: A POST endpoint `/greet/<name>` accepting a JSON payload with a message, returning a
greeting with the name from the URL and the message from the payload.

- **greet_params_only**: A GET endpoint `/greet-params/<name>` taking a URL parameter, returning a greeting with
the name from the URL.

- **greet_request_body_only**: A POST endpoint `/greet-body` accepting a JSON payload with a name and message,
returning a greeting with both.

OpenAPI specs for these endpoints are in `openapi_greeting_service.yml` in `test/test_files` directory.
"""


class GreetBody(BaseModel):
    message: str
    name: str


class MessageBody(BaseModel):
    message: str


# FastAPI app definitions
def create_greet_mix_params_body_app() -> FastAPI:
    app = FastAPI()

    @app.post("/greet/{name}")
    def greet(name: str, body: MessageBody):
        greeting = f"{body.message}, {name} from mix_params_body!"
        return JSONResponse(content={"greeting": greeting})

    return app


def create_greet_params_only_app() -> FastAPI:
    app = FastAPI()

    @app.get("/greet-params/{name}")
    def greet_params(name: str):
        greeting = f"Hello, {name} from params_only!"
        return JSONResponse(content={"greeting": greeting})

    return app


def create_greet_request_body_only_app() -> FastAPI:
    app = FastAPI()

    @app.post("/greet-body")
    def greet_request_body(body: GreetBody):
        greeting = f"{body.message}, {body.name} from request_body_only!"
        return JSONResponse(content={"greeting": greeting})

    return app


class TestOpenAPI:

    def test_greet_mix_params_body(self, test_files_path):
        builder = ClientConfigurationBuilder()
        config = (
            builder.with_openapi_spec(test_files_path / "yaml" / "openapi_greeting_service.yml")
            .with_http_client(FastAPITestClient(create_greet_mix_params_body_app()))
            .build()
        )
        client = OpenAPIServiceClient(config)
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": '{"name": "John", "message": "Bonjour"}',
                "name": "greet",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {"greeting": "Bonjour, John from mix_params_body!"}

    def test_greet_params_only(self, test_files_path):
        builder = ClientConfigurationBuilder()
        config = (
            builder.with_openapi_spec(test_files_path / "yaml" / "openapi_greeting_service.yml")
            .with_http_client(FastAPITestClient(create_greet_params_only_app()))
            .build()
        )
        client = OpenAPIServiceClient(config)
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": '{"name": "John"}',
                "name": "greetParams",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {"greeting": "Hello, John from params_only!"}

    def test_greet_request_body_only(self, test_files_path):
        builder = ClientConfigurationBuilder()
        config = (
            builder.with_openapi_spec(test_files_path / "yaml" / "openapi_greeting_service.yml")
            .with_http_client(FastAPITestClient(create_greet_request_body_only_app()))
            .build()
        )
        client = OpenAPIServiceClient(config)
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": '{"name": "John", "message": "Hola"}',
                "name": "greetBody",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {"greeting": "Hola, John from request_body_only!"}
