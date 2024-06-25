# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import (
    APIKeyCookie,
    APIKeyHeader,
    APIKeyQuery,
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)

from haystack_experimental.components.tools.openapi._openapi import OpenAPIServiceClient, ClientConfiguration
from test.components.tools.openapi.conftest import FastAPITestClient, create_openapi_spec

API_KEY = "secret_api_key"
BASIC_AUTH_USERNAME = "admin"
BASIC_AUTH_PASSWORD = "secret_password"

API_KEY_QUERY = "secret_api_key_query"
API_KEY_COOKIE = "secret_api_key_cookie"
BEARER_TOKEN = "secret_bearer_token"

OAUTH_TOKEN = "secret-oauth-token"

api_key_query = APIKeyQuery(name="api_key")
api_key_cookie = APIKeyCookie(name="api_key")
bearer_auth = HTTPBearer()

api_key_header = APIKeyHeader(name="X-API-Key")
basic_auth_http = HTTPBasic()


def create_greet_api_key_query_app() -> FastAPI:
    app = FastAPI()

    def api_key_query_auth(api_key: str = Depends(api_key_query)):
        if api_key != API_KEY_QUERY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return api_key

    @app.get("/greet-api-key-query/{name}")
    def greet_api_key_query(name: str, api_key: str = Depends(api_key_query_auth)):
        greeting = f"Hello, {name} from api_key_query_auth, using {api_key}"
        return JSONResponse(content={"greeting": greeting})

    return app


def create_greet_api_key_cookie_app() -> FastAPI:
    app = FastAPI()

    def api_key_cookie_auth(api_key: str = Depends(api_key_cookie)):
        if api_key != API_KEY_COOKIE:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return api_key

    @app.get("/greet-api-key-cookie/{name}")
    def greet_api_key_cookie(name: str, api_key: str = Depends(api_key_cookie_auth)):
        greeting = f"Hello, {name} from api_key_cookie_auth, using {api_key}"
        return JSONResponse(content={"greeting": greeting})

    return app


def create_greet_bearer_auth_app() -> FastAPI:
    app = FastAPI()

    def bearer_auth_scheme(
            credentials: HTTPAuthorizationCredentials = Depends(bearer_auth),  # noqa: B008
    ):
        if credentials.scheme != "Bearer" or credentials.credentials != BEARER_TOKEN:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return credentials.credentials

    @app.get("/greet-bearer-auth/{name}")
    def greet_bearer_auth(name: str, token: str = Depends(bearer_auth_scheme)):
        greeting = f"Hello, {name} from bearer_auth, using {token}"
        return JSONResponse(content={"greeting": greeting})

    return app


def create_greet_api_key_auth_app() -> FastAPI:
    app = FastAPI()

    def api_key_auth(api_key: str = Depends(api_key_header)):
        if api_key != API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
        return api_key

    @app.get("/greet-api-key/{name}")
    def greet_api_key(name: str, api_key: str = Depends(api_key_auth)):
        greeting = f"Hello, {name} from api_key_auth, using {api_key}"
        return JSONResponse(content={"greeting": greeting})

    return app


def create_greet_basic_auth_app() -> FastAPI:
    app = FastAPI()

    def basic_auth(credentials: HTTPBasicCredentials = Depends(basic_auth_http)):  # noqa: B008
        if credentials.username != BASIC_AUTH_USERNAME or credentials.password != BASIC_AUTH_PASSWORD:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return credentials.username

    @app.get("/greet-basic-auth/{name}")
    def greet_basic_auth(name: str, username: str = Depends(basic_auth)):
        greeting = f"Hello, {name} from basic_auth, using {username}"
        return JSONResponse(content={"greeting": greeting})

    return app


def create_greet_oauth_auth_app() -> FastAPI:
    app = FastAPI()

    def oauth_auth(token: HTTPAuthorizationCredentials = Depends(HTTPBearer())):  # noqa: B008
        if token.credentials != OAUTH_TOKEN:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return token

    @app.get("/greet-oauth/{name}")
    def greet_oauth(name: str, token: HTTPAuthorizationCredentials = Depends(oauth_auth)):  # noqa: B008
        greeting = f"Hello, {name} from oauth_auth, using {token}"
        return JSONResponse(content={"greeting": greeting})

    return app


class TestOpenAPIAuth:

    def test_greet_api_key_auth(self, test_files_path):
        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_greeting_service.yml"),
                                     request_sender=FastAPITestClient(create_greet_api_key_auth_app()),
                                     credentials=API_KEY)
        client = OpenAPIServiceClient(config)
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": '{"name": "John"}',
                "name": "greetApiKey",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {"greeting": "Hello, John from api_key_auth, using secret_api_key"}

    def test_greet_api_key_query_auth(self, test_files_path):
        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_greeting_service.yml"),
                                     request_sender=FastAPITestClient(create_greet_api_key_query_app()),
                                     credentials=API_KEY_QUERY)
        client = OpenAPIServiceClient(config)
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": '{"name": "John"}',
                "name": "greetApiKeyQuery",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {"greeting": "Hello, John from api_key_query_auth, using secret_api_key_query"}

    def test_greet_api_key_cookie_auth(self, test_files_path):

        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "openapi_greeting_service.yml"),
                                     request_sender=FastAPITestClient(create_greet_api_key_cookie_app()),
                                     credentials=API_KEY_COOKIE)

        client = OpenAPIServiceClient(config)
        payload = {
            "id": "call_NJr1NBz2Th7iUWJpRIJZoJIA",
            "function": {
                "arguments": '{"name": "John"}',
                "name": "greetApiKeyCookie",
            },
            "type": "function",
        }
        response = client.invoke(payload)
        assert response == {"greeting": "Hello, John from api_key_cookie_auth, using secret_api_key_cookie"}