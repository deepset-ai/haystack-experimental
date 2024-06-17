# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

import requests
import yaml

from haystack_experimental.components.tools.openapi._payload_extraction import (
    create_function_payload_extractor,
)
from haystack_experimental.components.tools.openapi._schema_conversion import (
    anthropic_converter,
    cohere_converter,
    openai_converter,
)
from haystack_experimental.components.tools.openapi.types import LLMProvider

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


def is_valid_http_url(url: str) -> bool:
    """
    Check if a URL is a valid HTTP/HTTPS URL.

    :param url: The URL to check.
    :return: True if the URL is a valid HTTP/HTTPS URL, False otherwise.
    """
    r = urlparse(url)
    return all([r.scheme in ["http", "https"], r.netloc])


def send_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send an HTTP request and return the response.

    :param request: The request to send.
    :return: The response from the server.
    """
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
        logger.warning("HTTP error occurred: %s while sending request to %s", e, url)
        raise HttpClientError(f"HTTP error occurred: {e}") from e
    except requests.exceptions.RequestException as e:
        logger.warning("Request error occurred: %s while sending request to %s", e, url)
        raise HttpClientError(f"HTTP error occurred: {e}") from e
    except Exception as e:
        logger.warning("An error occurred: %s while sending request to %s", e, url)
        raise HttpClientError(f"An error occurred: {e}") from e


# Authentication strategies
def create_api_key_auth_function(api_key: str):
    """
    Create a function that applies the API key authentication strategy to a given request.

    :param api_key: the API key to use for authentication.
    :return: a function that applies the API key authentication to a request
    at the schema specified location.
    """

    def apply_auth(security_scheme: Dict[str, Any], request: Dict[str, Any]):
        """
        Apply the API key authentication strategy to the given request.

        :param security_scheme: the security scheme from the OpenAPI spec.
        :param request: the request to apply the authentication to.
        """
        if security_scheme["in"] == "header":
            request.setdefault("headers", {})[security_scheme["name"]] = api_key
        elif security_scheme["in"] == "query":
            request.setdefault("params", {})[security_scheme["name"]] = api_key
        elif security_scheme["in"] == "cookie":
            request.setdefault("cookies", {})[security_scheme["name"]] = api_key
        else:
            raise ValueError(
                f"Unsupported apiKey authentication location: {security_scheme['in']}, "
                f"must be one of 'header', 'query', or 'cookie'"
            )

    return apply_auth


def create_http_auth_function(token: str):
    """
    Create a function that applies the http authentication strategy to a given request.

    :param token: the authentication token to use.
    :return: a function that applies the API key authentication to a request
    at the schema specified location.
    """

    def apply_auth(security_scheme: Dict[str, Any], request: Dict[str, Any]):
        """
        Apply the HTTP authentication strategy to the given request.

        :param security_scheme: the security scheme from the OpenAPI spec.
        :param request: the request to apply the authentication to.
        """
        if security_scheme["type"] == "http":
            # support bearer http auth, no basic support yet
            if security_scheme["scheme"].lower() == "bearer":
                if not token:
                    raise ValueError("Token must be provided for Bearer Auth.")
                request.setdefault("headers", {})[
                    "Authorization"
                ] = f"Bearer {token}"
            else:
                raise ValueError(
                    f"Unsupported HTTP authentication scheme: {security_scheme['scheme']}"
                )
        else:
            raise ValueError(
                "HTTPAuthentication strategy received a non-HTTP security scheme."
            )

    return apply_auth


class HttpClientError(Exception):
    """Exception raised for errors in the HTTP client."""


@dataclass
class Operation:
    """
     Represents an operation in an OpenAPI specification

     See https://spec.openapis.org/oas/latest.html#paths-object for details.
     Path objects can contain multiple operations, each with a unique combination of path and method.

     Attributes:
        path (str): Path of the operation.
        method (str): HTTP method of the operation.
        operation_dict (Dict[str, Any]): Operation details from OpenAPI spec.
        spec_dict (Dict[str, Any]): The encompassing OpenAPI specification.
        security_requirements (List[Dict[str, List[str]]]): Security requirements for the operation.
        request_body (Dict[str, Any]): Request body details.
        parameters (List[Dict[str, Any]]): Parameters for the operation.
    """

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

        :param location: The location of the parameters to get.
        :return: The parameters for the operation as a list of dictionaries.
        """
        if location:
            return [param for param in self.parameters if param["in"] == location]
        return self.parameters

    def get_server(self, server_index: int = 0) -> str:
        """
        Get the servers for the operation.

        :param server_index: The index of the server to use.
        :return: The server URL.
        :raises ValueError: If no servers are found in the specification.
        """
        servers = self.operation_dict.get("servers", []) or self.spec_dict.get(
            "servers", []
        )
        if not servers:
            raise ValueError("No servers found in the provided specification.")
        if server_index >= len(servers):
            raise ValueError(
                f"Server index {server_index} is out of bounds. "
                f"Only {len(servers)} servers found."
            )
        return servers[server_index].get(
            "url"
        )  # just use the first server from the list


class OpenAPISpecification:
    """
    Represents an OpenAPI specification. See https://spec.openapis.org/oas/latest.html for details.
    """

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
    def from_str(cls, content: str) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a string.

        :param content: The string content of the OpenAPI specification.
        :return: The OpenAPISpecification instance.
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

        :param spec_file: The file path to the OpenAPI specification.
        :return: The OpenAPISpecification instance.
        """
        with open(spec_file, encoding="utf-8") as file:
            content = file.read()
        return cls.from_str(content)

    @classmethod
    def from_url(cls, url: str) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a URL.

        :param url: The URL to fetch the OpenAPI specification from.
        :return: The OpenAPISpecification instance.
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
        Find an Operation by operationId.

        :param op_id: The operationId of the operation.
        :param method: The HTTP method of the operation.
        :return: The matching operation
        :raises ValueError: If no operation is found with the given operationId.
        """
        for path, path_item in self.spec_dict.get("paths", {}).items():
            op: Operation = self.get_operation_item(path, path_item, method)
            if op_id in op.operation_dict.get("operationId", ""):
                return self.get_operation_item(path, path_item, method)
        raise ValueError(
            f"No operation found with operationId {op_id}, method {method}"
        )

    def get_operation_item(
        self, path: str, path_item: Dict[str, Any], method: Optional[str] = None
    ) -> Operation:
        """
        Gets a particular Operation item from the OpenAPI specification given the path and method.

        :param path: The path of the operation.
        :param path_item: The path item from the OpenAPI specification.
        :param method: The HTTP method of the operation.
        :return: The operation
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
        raise ValueError(f"No operations found at path {path} and method {method}")

    def get_security_schemes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the security schemes from the OpenAPI specification.

        :return: The security schemes as a dictionary.
        """
        return self.spec_dict.get("components", {}).get("securitySchemes", {})


class ClientConfiguration:
    """Configuration for the OpenAPI client."""

    def __init__(  # noqa: PLR0913 pylint: disable=too-many-arguments
        self,
        openapi_spec: Union[str, Path],
        credentials: Optional[str] = None,
        request_sender: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        llm_provider: Optional[LLMProvider] = None,
    ):  # noqa: PLR0913
        if isinstance(openapi_spec, (str, Path)) and os.path.isfile(openapi_spec):
            self.openapi_spec = OpenAPISpecification.from_file(openapi_spec)
        elif isinstance(openapi_spec, str):
            if is_valid_http_url(openapi_spec):
                self.openapi_spec = OpenAPISpecification.from_url(openapi_spec)
            else:
                self.openapi_spec = OpenAPISpecification.from_str(openapi_spec)
        else:
            raise ValueError(
                "Invalid OpenAPI specification format. Expected file path or dictionary."
            )

        self.credentials = credentials
        self.request_sender = request_sender or send_request
        self.llm_provider: LLMProvider = llm_provider or LLMProvider.OPENAI

    def get_auth_function(self) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
        """
        Get the authentication function that sets a schema specified authentication to the request.

        The function takes a security scheme and a request as arguments:
            `security_scheme: Dict[str, Any] - The security scheme from the OpenAPI spec.`
            `request: Dict[str, Any] - The request to apply the authentication to.`
        :return: The authentication function.
        """
        security_schemes = self.openapi_spec.get_security_schemes()
        if not self.credentials:
            return lambda security_scheme, request: None  # No-op function
        if isinstance(self.credentials, str):
            return self._create_authentication_from_string(
                self.credentials, security_schemes
            )
        raise ValueError(f"Unsupported credentials type: {type(self.credentials)}")

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the tools definitions used as tools LLM parameter.

        :return: The tools definitions passed to the LLM as tools parameter.
        """
        provider_to_converter = {
            "anthropic": anthropic_converter,
            "cohere": cohere_converter,
        }
        converter = provider_to_converter.get(self.llm_provider.value, openai_converter)
        return converter(self.openapi_spec)

    def get_payload_extractor(self):
        """
        Get the payload extractor for the LLM provider.

        This function knows how to extract the exact function payload from the LLM generated function calling payload.
        :return: The payload extractor function.
        """
        provider_to_arguments_field_name = {
            "anthropic": "input",
            "cohere": "parameters",
        }  # add more providers here
        # default to OpenAI "arguments"
        arguments_field_name = provider_to_arguments_field_name.get(
            self.llm_provider.value, "arguments"
        )
        return create_function_payload_extractor(arguments_field_name)

    def _create_authentication_from_string(
        self, credentials: str, security_schemes: Dict[str, Any]
    ) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
        for scheme in security_schemes.values():
            if scheme["type"] == "apiKey":
                return create_api_key_auth_function(api_key=credentials)
            if scheme["type"] == "http":
                return create_http_auth_function(token=credentials)
            if scheme["type"] == "oauth2":
                raise NotImplementedError("OAuth2 authentication is not yet supported.")
        raise ValueError(
            f"Unable to create authentication from provided credentials: {credentials}"
        )


def build_request(operation: Operation, **kwargs) -> Dict[str, Any]:
    """
    Build an HTTP request for the operation.

    :param operation: The operation to build the request for.
    :param kwargs: The arguments to use for building the request.
    :return: The HTTP request as a dictionary.
    """
    path = operation.path
    for parameter in operation.get_parameters("path"):
        param_value = kwargs.get(parameter["name"], None)
        if param_value:
            path = path.replace(f"{{{parameter['name']}}}", str(param_value))
        elif parameter.get("required", False):
            raise ValueError(f"Missing required path parameter: {parameter['name']}")
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
            raise ValueError(f"Missing required header parameter: {parameter['name']}")
    # query params
    query_params = {}
    for parameter in operation.get_parameters("query"):
        param_value = kwargs.get(parameter["name"], None)
        if param_value:
            query_params[parameter["name"]] = param_value
        elif parameter.get("required", False):
            raise ValueError(f"Missing required query parameter: {parameter['name']}")

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


def apply_authentication(
    auth_strategy: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    operation: Operation,
    request: Dict[str, Any],
):
    """
    Apply the authentication strategy to the given request.

    :param auth_strategy: The authentication strategy to apply.
    This is a function that takes a security scheme and a request as arguments (at runtime)
    and applies the authentication
    :param operation: The operation to apply the authentication to.
    :param request: The request to apply the authentication to.
    """
    security_requirements = operation.security_requirements
    security_schemes = operation.spec_dict.get("components", {}).get(
        "securitySchemes", {}
    )
    if security_requirements:
        for requirement in security_requirements:
            for scheme_name in requirement:
                if scheme_name in security_schemes:
                    security_scheme = security_schemes[scheme_name]
                    auth_strategy(security_scheme, request)
                break


class OpenAPIServiceClient:
    """
    A client for invoking operations on REST services defined by OpenAPI specifications.
    """

    def __init__(self, client_config: ClientConfiguration):
        self.client_config = client_config
        self.request_sender = client_config.request_sender

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
        operation = self.client_config.openapi_spec.find_operation_by_id(
            fn_invocation_payload.get("name")
        )
        request = build_request(operation, **fn_invocation_payload.get("arguments"))
        apply_authentication(self.client_config.get_auth_function(), operation, request)
        return self.request_sender(request)


class OpenAPIClientError(Exception):
    """Exception raised for errors in the OpenAPI client."""
