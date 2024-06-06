# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

import requests
import yaml

from haystack_experimental.components.tools.openapi.payload_extraction import (
    create_function_payload_extractor,
)
from haystack_experimental.components.tools.openapi.schema_conversion import (
    anthropic_converter,
    cohere_converter,
    openai_converter,
)

VALID_HTTP_METHODS = [
    "get",
    "put",
    "post",
    "delete",
    "options",
    "head",
    "patch",
    "trace",
]
MIN_REQUIRED_OPENAPI_SPEC_VERSION = 3
logger = logging.getLogger(__name__)


class AuthenticationStrategy:
    """
    Represents an authentication strategy that can be applied to an HTTP request.
    """

    def apply_auth(self, security_scheme: Dict[str, Any], request: Dict[str, Any]):
        """
        Apply the authentication strategy to the given request.

        :param security_scheme: the security scheme from the OpenAPI spec.
        :param request: the request to apply the authentication to.
        """


@dataclass
class ApiKeyAuthentication(AuthenticationStrategy):
    """API key authentication strategy."""

    api_key: Optional[str] = None

    def apply_auth(self, security_scheme: Dict[str, Any], request: Dict[str, Any]):
        """
        Apply the API key authentication strategy to the given request.

        :param security_scheme: the security scheme from the OpenAPI spec.
        :param request: the request to apply the authentication to.
        """
        if security_scheme["in"] == "header":
            request.setdefault("headers", {})[security_scheme["name"]] = self.api_key
        elif security_scheme["in"] == "query":
            request.setdefault("params", {})[security_scheme["name"]] = self.api_key
        elif security_scheme["in"] == "cookie":
            request.setdefault("cookies", {})[security_scheme["name"]] = self.api_key
        else:
            raise ValueError(
                f"Unsupported apiKey authentication location: {security_scheme['in']}, "
                f"must be one of 'header', 'query', or 'cookie'"
            )


@dataclass
class HTTPAuthentication(AuthenticationStrategy):
    """HTTP authentication strategy."""

    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None

    def __post_init__(self):
        if not self.token and (not self.username or not self.password):
            raise ValueError(
                "For HTTP Basic Auth, both username and password must be provided. "
                "For Bearer Auth, a token must be provided."
            )

    def apply_auth(self, security_scheme: Dict[str, Any], request: Dict[str, Any]):
        """
        Apply the HTTP authentication strategy to the given request.

        :param security_scheme: the security scheme from the OpenAPI spec.
        :param request: the request to apply the authentication to.
        """
        if security_scheme["type"] == "http":
            if security_scheme["scheme"].lower() == "basic":
                if not self.username or not self.password:
                    raise ValueError(
                        "Username and password must be provided for Basic Auth."
                    )
                credentials = f"{self.username}:{self.password}"
                encoded_credentials = b64encode(credentials.encode("utf-8")).decode(
                    "utf-8"
                )
                request.setdefault("headers", {})[
                    "Authorization"
                ] = f"Basic {encoded_credentials}"
            elif security_scheme["scheme"].lower() == "bearer":
                if not self.token:
                    raise ValueError("Token must be provided for Bearer Auth.")
                request.setdefault("headers", {})[
                    "Authorization"
                ] = f"Bearer {self.token}"
            else:
                raise ValueError(
                    f"Unsupported HTTP authentication scheme: {security_scheme['scheme']}"
                )
        else:
            raise ValueError(
                "HTTPAuthentication strategy received a non-HTTP security scheme."
            )


class HttpClientError(Exception):
    """Exception raised for errors in the HTTP client."""


@dataclass
class Operation:
    """Represents an operation in an OpenAPI specification."""

    path: str
    method: str
    operation_dict: Dict[str, Any]
    spec_dict: Dict[str, Any]
    security_requirements: List[Dict[str, List[str]]] = field(init=False)
    request_body: Dict[str, Any] = field(init=False)
    parameters: List[Dict[str, Any]] = field(init=False)

    def __post_init__(self):
        if self.method.lower() not in VALID_HTTP_METHODS:
            raise ValueError(f"Invalid HTTP method: {self.method}")
        self.method = self.method.lower()
        self.security_requirements = self.operation_dict.get(
            "security", []
        ) or self.spec_dict.get("security", [])
        self.request_body = self.operation_dict.get("requestBody", {})
        self.parameters = self.operation_dict.get(
            "parameters", []
        ) + self.spec_dict.get("paths", {}).get(self.path, {}).get("parameters", [])

    def get_parameters(
        self, location: Optional[Literal["header", "query", "path"]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the parameters for the operation.
        """
        if location:
            return [param for param in self.parameters if param["in"] == location]
        return self.parameters

    def get_server(self) -> str:
        """
        Get the servers for the operation.
        """
        servers = self.operation_dict.get("servers", []) or self.spec_dict.get(
            "servers", []
        )
        return servers[0].get("url", "")  # just use the first server from the list


class OpenAPISpecification:
    """Represents an OpenAPI specification."""

    def __init__(self, spec_dict: Dict[str, Any]):
        if not isinstance(spec_dict, Dict):
            raise ValueError(
                f"Invalid OpenAPI specification, expected a dictionary: {spec_dict}"
            )
        # just a crude sanity check, by no means a full validation
        if (
            "openapi" not in spec_dict
            or "paths" not in spec_dict
            or "servers" not in spec_dict
        ):
            raise ValueError(
                "Invalid OpenAPI specification format. See https://swagger.io/specification/ for details.",
                spec_dict,
            )
        self.spec_dict = spec_dict

    @classmethod
    def from_dict(cls, spec_dict: Dict[str, Any]) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a dictionary.
        """
        parser = cls(spec_dict)
        return parser

    @classmethod
    def from_str(cls, content: str) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a string.
        """
        try:
            loaded_spec = json.loads(content)
        except json.JSONDecodeError:
            try:
                loaded_spec = yaml.safe_load(content)
            except yaml.YAMLError as e:
                raise ValueError(
                    "Content cannot be decoded as JSON or YAML: " + str(e)
                ) from e
        return cls(loaded_spec)

    @classmethod
    def from_file(cls, spec_file: Union[str, Path]) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a file.
        """
        with open(spec_file, encoding="utf-8") as file:
            content = file.read()
        return cls.from_str(content)

    @classmethod
    def from_url(cls, url: str) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a URL.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content = response.text
        except requests.RequestException as e:
            raise ConnectionError(
                f"Failed to fetch the specification from URL: {url}. {e!s}"
            ) from e
        return cls.from_str(content)

    def find_operation_by_id(
        self, op_id: str, method: Optional[str] = None
    ) -> Operation:
        """
        Find an operation by operationId.
        """
        for path, path_item in self.spec_dict.get("paths", {}).items():
            op: Operation = self.get_operation_item(path, path_item, method)
            if op_id in op.operation_dict.get("operationId", ""):
                return self.get_operation_item(path, path_item, method)
        raise ValueError(f"No operation found with operationId {op_id}")

    def get_operation_item(
        self, path: str, path_item: Dict[str, Any], method: Optional[str] = None
    ) -> Operation:
        """
        Get an operation item from the OpenAPI specification.

        :param path: The path of the operation.
        :param path_item: The path item from the OpenAPI specification.
        :param method: The HTTP method of the operation.
        """
        if method:
            operation_dict = path_item.get(method.lower(), {})
            if not operation_dict:
                raise ValueError(
                    f"No operation found for method {method} at path {path}"
                )
            return Operation(path, method.lower(), operation_dict, self.spec_dict)
        if len(path_item) == 1:
            method, operation_dict = next(iter(path_item.items()))
            return Operation(path, method, operation_dict, self.spec_dict)
        if len(path_item) > 1:
            raise ValueError(
                f"Multiple operations found at path {path}, method parameter is required."
            )
        raise ValueError(f"No operations found at path {path}.")

    def get_security_schemes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the security schemes from the OpenAPI specification.
        """
        return self.spec_dict.get("components", {}).get("securitySchemes", {})


class ClientConfiguration:
    """Configuration for the OpenAPI client."""

    def __init__(  # noqa: PLR0913 pylint: disable=too-many-arguments
        self,
        openapi_spec: Union[str, Path, Dict[str, Any]],
        credentials: Optional[
            Union[str, Dict[str, Any], AuthenticationStrategy]
        ] = None,
        request_sender: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        llm_provider: Optional[str] = None,
    ):  # noqa: PLR0913
        if isinstance(openapi_spec, (str, Path)) and os.path.isfile(openapi_spec):
            self.openapi_spec = OpenAPISpecification.from_file(openapi_spec)
        elif isinstance(openapi_spec, dict):
            self.openapi_spec = OpenAPISpecification.from_dict(openapi_spec)
        elif isinstance(openapi_spec, str):
            if self.is_valid_http_url(openapi_spec):
                self.openapi_spec = OpenAPISpecification.from_url(openapi_spec)
            else:
                self.openapi_spec = OpenAPISpecification.from_str(openapi_spec)
        else:
            raise ValueError(
                "Invalid OpenAPI specification format. Expected file path or dictionary."
            )

        self.credentials = credentials
        self.request_sender = request_sender
        self.llm_provider = llm_provider or "openai"

    def get_auth_config(self) -> AuthenticationStrategy:
        """
        Get the authentication configuration.
        """
        if not self.credentials:
            return AuthenticationStrategy()
        if isinstance(self.credentials, AuthenticationStrategy):
            return self.credentials
        security_schemes = self.openapi_spec.get_security_schemes()
        if isinstance(self.credentials, str):
            return self._create_authentication_from_string(
                self.credentials, security_schemes
            )
        if isinstance(self.credentials, dict):
            return self._create_authentication_from_dict(self.credentials)
        raise ValueError(f"Unsupported credentials type: {type(self.credentials)}")

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the tools definitions used as tools LLM parameter.
        """
        provider_to_converter = {
            "anthropic": anthropic_converter,
            "cohere": cohere_converter,
        }
        converter = provider_to_converter.get(self.llm_provider, openai_converter)
        return converter(self.openapi_spec)

    def get_payload_extractor(self):
        """
        Get the payload extractor for the LLM provider.
        """
        provider_to_arguments_field_name = {
            "anthropic": "input",
            "cohere": "parameters",
        }  # add more providers here
        # default to OpenAI "arguments"
        arguments_field_name = provider_to_arguments_field_name.get(
            self.llm_provider, "arguments"
        )
        return create_function_payload_extractor(arguments_field_name)

    def _create_authentication_from_string(
        self, credentials: str, security_schemes: Dict[str, Any]
    ) -> AuthenticationStrategy:
        for scheme in security_schemes.values():
            if scheme["type"] == "apiKey":
                return ApiKeyAuthentication(api_key=credentials)
            if scheme["type"] == "http":
                return HTTPAuthentication(token=credentials)
            if scheme["type"] == "oauth2":
                raise NotImplementedError("OAuth2 authentication is not yet supported.")
        raise ValueError(
            f"Unable to create authentication from provided credentials: {credentials}"
        )

    def _create_authentication_from_dict(
        self, credentials: Dict[str, Any]
    ) -> AuthenticationStrategy:
        if "username" in credentials and "password" in credentials:
            return HTTPAuthentication(
                username=credentials["username"], password=credentials["password"]
            )
        if "api_key" in credentials:
            return ApiKeyAuthentication(api_key=credentials["api_key"])
        if "token" in credentials:
            return HTTPAuthentication(token=credentials["token"])
        if "access_token" in credentials:
            raise NotImplementedError("OAuth2 authentication is not yet supported.")
        raise ValueError(
            "Unable to create authentication from provided credentials: {credentials}"
        )

    def is_valid_http_url(self, url: str) -> bool:
        """Check if a URL is a valid HTTP/HTTPS URL."""
        r = urlparse(url)
        return all([r.scheme in ["http", "https"], r.netloc])


class OpenAPIServiceClient:
    """
    A client for invoking operations on REST services defined by OpenAPI specifications.
    """

    def __init__(self, client_config: ClientConfiguration):
        self.client_config = client_config
        self.request_sender = client_config.request_sender or self._request_sender()

    def invoke(self, function_payload: Any) -> Any:
        """
        Invokes a function specified in the function payload.

        :param function_payload: The function payload containing the details of the function to be invoked.
        :returns: The response from the service after invoking the function.
        :raises OpenAPIClientError: If the function invocation payload cannot be extracted from the function payload.
        :raises HttpClientError: If an error occurs while sending the request and receiving the response.
        """
        fn_extractor = self.client_config.get_payload_extractor()
        fn_invocation_payload = fn_extractor(function_payload)
        if not fn_invocation_payload:
            raise OpenAPIClientError(
                f"Failed to extract function invocation payload from {function_payload}"
            )
        # fn_invocation_payload, if not empty, guaranteed to have "name" and "arguments" keys from here on
        operation = self.client_config.openapi_spec.find_operation_by_id(fn_invocation_payload.get("name"))
        request = self._build_request(operation, **fn_invocation_payload.get("arguments"))
        self._apply_authentication(self.client_config.get_auth_config(), operation, request)
        return self.request_sender(request)

    def _request_sender(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Returns a callable that sends the request using the HTTP client.
        """

        def send_request(request: Dict[str, Any]) -> Dict[str, Any]:
            url = request["url"]
            headers = {**request.get("headers", {})}
            try:
                response = requests.request(
                    request["method"],
                    url,
                    headers=headers,
                    params=request.get("params", {}),
                    json=request.get("json"),
                    auth=request.get("auth"),
                    timeout=10,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                logger.warning(
                    "HTTP error occurred: %s while sending request to %s", e, url
                )
                raise HttpClientError(f"HTTP error occurred: {e}") from e
            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Request error occurred: %s while sending request to %s", e, url
                )
                raise HttpClientError(f"HTTP error occurred: {e}") from e
            except Exception as e:
                logger.warning(
                    "An error occurred: %s while sending request to %s", e, url
                )
                raise HttpClientError(f"An error occurred: {e}") from e

        return send_request

    def _build_request(self, operation: Operation, **kwargs) -> Any:
        # url
        path = operation.path
        for parameter in operation.get_parameters("path"):
            param_value = kwargs.get(parameter["name"], None)
            if param_value:
                path = path.replace(f"{{{parameter['name']}}}", str(param_value))
            elif parameter.get("required", False):
                raise ValueError(
                    f"Missing required path parameter: {parameter['name']}"
                )
        url = operation.get_server() + path
        # method
        method = operation.method.lower()
        # headers
        headers = {}
        for parameter in operation.get_parameters("header"):
            param_value = kwargs.get(parameter["name"], None)
            if param_value:
                headers[parameter["name"]] = str(param_value)
            elif parameter.get("required", False):
                raise ValueError(
                    f"Missing required header parameter: {parameter['name']}"
                )
        # query params
        query_params = {}
        for parameter in operation.get_parameters("query"):
            param_value = kwargs.get(parameter["name"], None)
            if param_value:
                query_params[parameter["name"]] = param_value
            elif parameter.get("required", False):
                raise ValueError(
                    f"Missing required query parameter: {parameter['name']}"
                )

        json_payload = None
        request_body = operation.request_body
        if request_body:
            content = request_body.get("content", {})
            if "application/json" in content:
                json_payload = {**kwargs}
            else:
                raise NotImplementedError("Request body content type not supported")
        return {
            "url": url,
            "method": method,
            "headers": headers,
            "params": query_params,
            "json": json_payload,
        }

    def _apply_authentication(
        self,
        auth: AuthenticationStrategy,
        operation: Operation,
        request: Dict[str, Any],
    ):
        auth_config = auth or AuthenticationStrategy()
        security_requirements = operation.security_requirements
        security_schemes = operation.spec_dict.get("components", {}).get(
            "securitySchemes", {}
        )
        if security_requirements:
            for requirement in security_requirements:
                for scheme_name in requirement:
                    if scheme_name in security_schemes:
                        security_scheme = security_schemes[scheme_name]
                        auth_config.apply_auth(security_scheme, request)
                    break


class OpenAPIClientError(Exception):
    """Exception raised for errors in the OpenAPI client."""
