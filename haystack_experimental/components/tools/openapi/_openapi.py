# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import requests

from haystack_experimental.components.tools.openapi._payload_extraction import (
    create_function_payload_extractor,
)
from haystack_experimental.components.tools.openapi._schema_conversion import (
    anthropic_converter,
    cohere_converter,
    openai_converter,
)
from haystack_experimental.components.tools.openapi.types import LLMProvider, OpenAPISpecification, Operation
from haystack_experimental.components.tools.utils import normalize_tool_definition

MIN_REQUIRED_OPENAPI_SPEC_VERSION = 3
logger = logging.getLogger(__name__)


def send_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send an HTTP request and return the response.

    :param request: The request to send.
    :returns: The response from the server.
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
            timeout=30,
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
def create_api_key_auth_function(api_key: str) -> Callable[[Dict[str, Any], Dict[str, Any]], None]:
    """
    Create a function that applies the API key authentication strategy to a given request.

    :param api_key: the API key to use for authentication.
    :returns: a function that applies the API key authentication to a request
    at the schema specified location.
    """

    def apply_auth(security_scheme: Dict[str, Any], request: Dict[str, Any]) -> None:
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


def create_http_auth_function(token: str) -> Callable[[Dict[str, Any], Dict[str, Any]], None]:
    """
    Create a function that applies the http authentication strategy to a given request.

    :param token: the authentication token to use.
    :returns: a function that applies the API key authentication to a request
    at the schema specified location.
    """

    def apply_auth(security_scheme: Dict[str, Any], request: Dict[str, Any]) -> None:
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
                request.setdefault("headers", {})["Authorization"] = f"Bearer {token}"
            else:
                raise ValueError(f"Unsupported HTTP authentication scheme: {security_scheme['scheme']}")
        else:
            raise ValueError("HTTPAuthentication strategy received a non-HTTP security scheme.")

    return apply_auth


class HttpClientError(Exception):
    """Exception raised for errors in the HTTP client."""


class ClientConfiguration:
    """Configuration for the OpenAPI client."""

    def __init__(
        self,
        openapi_spec: OpenAPISpecification,
        credentials: Optional[str] = None,
        request_sender: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        llm_provider: LLMProvider = LLMProvider.OPENAI,
        operations_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):  # noqa: PLR0913  # pylint: disable=too-many-positional-arguments
        """
        Initialize a ClientConfiguration instance.

        :param openapi_spec: The OpenAPI specification to use for the client.
        :param credentials: The credentials to use for authentication.
        :param request_sender: The function to use for sending requests.
        :param llm_provider: The LLM provider to use for generating tools definitions.
        :param operations_filter: A function to filter the functions to register with LLMs.
        :raises ValueError: If the OpenAPI specification format is invalid.
        """
        self.openapi_spec = openapi_spec
        self.credentials = credentials
        self.request_sender = request_sender or send_request
        self.llm_provider: LLMProvider = llm_provider
        self.operation_filter = operations_filter

    def get_auth_function(self) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
        """
        Get the authentication function that sets a schema specified authentication to the request.

        The function takes a security scheme and a request as arguments:
            `security_scheme: Dict[str, Any] - The security scheme from the OpenAPI spec.`
            `request: Dict[str, Any] - The request to apply the authentication to.`
        :returns: The authentication function.
        :raises ValueError: If the credentials type is not supported.
        """
        security_schemes = self.openapi_spec.get_security_schemes()
        if not self.credentials:
            return lambda security_scheme, request: None  # No-op function
        if isinstance(self.credentials, str):
            return self._create_authentication_from_string(self.credentials, security_schemes)
        raise ValueError(f"Unsupported credentials type: {type(self.credentials)}")

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the tools definitions used as tools LLM parameter.

        :returns: The tools definitions ready to be passed to the LLM as tools parameter.
        """
        provider_to_converter = defaultdict(
            lambda: openai_converter,
            {
                LLMProvider.ANTHROPIC: anthropic_converter,
                LLMProvider.COHERE: cohere_converter,
            },
        )
        converter = provider_to_converter[self.llm_provider]
        tools_definitions = converter(self.openapi_spec, self.operation_filter)
        return [normalize_tool_definition(t) for t in tools_definitions]

    def get_payload_extractor(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """
        Get the payload extractor for the LLM provider.

        This function knows how to extract the exact function payload from the LLM generated function calling payload.
        :returns: The payload extractor function.
        """
        provider_to_arguments_field_name = defaultdict(
            lambda: "arguments",
            {
                LLMProvider.ANTHROPIC: "input",
                LLMProvider.COHERE: "parameters",
            },
        )
        arguments_field_name = provider_to_arguments_field_name[self.llm_provider]
        return create_function_payload_extractor(arguments_field_name)

    def _create_authentication_from_string(
        self, credentials: str, security_schemes: Dict[str, Any]
    ) -> Callable[[Dict[str, Any], Dict[str, Any]], Any]:
        for scheme in security_schemes.values():
            if scheme["type"] == "apiKey":
                return create_api_key_auth_function(api_key=credentials)
            if scheme["type"] == "http":
                return create_http_auth_function(token=credentials)
            raise ValueError(f"Unsupported authentication type '{scheme['type']}' provided.")
        raise ValueError(f"Unable to create authentication from provided credentials: {credentials}")


def build_request(operation: Operation, **kwargs) -> Dict[str, Any]:
    """
    Build an HTTP request for the operation.

    :param operation: The operation to build the request for.
    :param kwargs: The arguments to use for building the request.
    :returns: The HTTP request as a dictionary.
    :raises ValueError: If a required parameter is missing.
    :raises NotImplementedError: If the request body content type is not supported. We only support JSON payloads.
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
    security_schemes = operation.spec_dict.get("components", {}).get("securitySchemes", {})
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

    def invoke(self, function_payload: Any) -> Any:
        """
        Invokes a function specified in the function payload.

        :param function_payload: The function payload containing the details of the function to be invoked.
        :returns: The response from the service after invoking the function.
        :raises OpenAPIClientError: If the function invocation payload cannot be extracted from the function payload.
        :raises HttpClientError: If an error occurs while sending the request and receiving the response.
        """
        fn_invocation_payload = {}
        try:
            fn_extractor = self.client_config.get_payload_extractor()
            fn_invocation_payload = fn_extractor(function_payload)
        except Exception as e:
            raise OpenAPIClientError(f"Error extracting function invocation payload: {str(e)}") from e

        if "name" not in fn_invocation_payload or "arguments" not in fn_invocation_payload:
            raise OpenAPIClientError(
                f"Function invocation payload does not contain 'name' or 'arguments' keys: {fn_invocation_payload}, "
                f"the payload extraction function may be incorrect."
            )
        # fn_invocation_payload, if not empty, guaranteed to have "name" and "arguments" keys from here on
        operation = self.client_config.openapi_spec.find_operation_by_id(fn_invocation_payload["name"])
        request = build_request(operation, **fn_invocation_payload["arguments"])
        apply_authentication(self.client_config.get_auth_function(), operation, request)
        return self.client_config.request_sender(request)


class OpenAPIClientError(Exception):
    """Exception raised for errors in the OpenAPI client."""
