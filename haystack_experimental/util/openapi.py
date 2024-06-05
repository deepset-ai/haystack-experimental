# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
import os
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

import jsonref
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3 import Retry

VALID_HTTP_METHODS = ["get", "put", "post", "delete", "options", "head", "patch", "trace"]

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
                    raise ValueError("Username and password must be provided for Basic Auth.")
                credentials = f"{self.username}:{self.password}"
                encoded_credentials = b64encode(credentials.encode("utf-8")).decode("utf-8")
                request.setdefault("headers", {})["Authorization"] = f"Basic {encoded_credentials}"
            elif security_scheme["scheme"].lower() == "bearer":
                if not self.token:
                    raise ValueError("Token must be provided for Bearer Auth.")
                request.setdefault("headers", {})["Authorization"] = f"Bearer {self.token}"
            else:
                raise ValueError(f"Unsupported HTTP authentication scheme: {security_scheme['scheme']}")
        else:
            raise ValueError("HTTPAuthentication strategy received a non-HTTP security scheme.")


@dataclass
class HttpClientConfig:
    """Configuration for the HTTP client."""

    timeout: int = 10
    max_retries: int = 3
    backoff_factor: float = 0.3
    retry_on_status: set = field(default_factory=lambda: {500, 502, 503, 504})
    default_headers: Dict[str, str] = field(default_factory=dict)


class HttpClient:
    """HTTP client for sending requests."""

    def __init__(self, config: Optional[HttpClientConfig] = None):
        self.config = config or HttpClientConfig()
        self.session = requests.Session()
        self._initialize_session()

    def _initialize_session(self) -> None:
        retries = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=self.config.retry_on_status,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update(self.config.default_headers)

    def send_request(self, request: Dict[str, Any]) -> Any:
        """
        Send an HTTP request using the provided request dictionary.

        :param request: A dictionary containing the request details.
        """
        url = request["url"]
        headers = {**self.config.default_headers, **request.get("headers", {})}
        try:
            response = self.session.request(
                request["method"],
                request["url"],
                headers=headers,
                params=request.get("params", {}),
                json=request.get("json"),
                auth=request.get("auth"),
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
        self.security_requirements = self.operation_dict.get("security", []) or self.spec_dict.get("security", [])
        self.request_body = self.operation_dict.get("requestBody", {})
        self.parameters = self.operation_dict.get("parameters", []) + self.spec_dict.get("paths", {}).get(
            self.path, {}
        ).get("parameters", [])

    def get_parameters(self, location: Optional[Literal["header", "query", "path"]] = None) -> List[Dict[str, Any]]:
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
        servers = self.operation_dict.get("servers", []) or self.spec_dict.get("servers", [])
        return servers[0].get("url", "")  # just use the first server from the list


class OpenAPISpecification:
    """Represents an OpenAPI specification."""

    def __init__(self, spec_dict: Dict[str, Any]):
        if not isinstance(spec_dict, Dict):
            raise ValueError(f"Invalid OpenAPI specification, expected a dictionary: {spec_dict}")
        # just a crude sanity check, by no means a full validation
        if "openapi" not in spec_dict or "paths" not in spec_dict or "servers" not in spec_dict:
            raise ValueError(
                "Invalid OpenAPI specification format. See https://swagger.io/specification/ for details.", spec_dict
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
                raise ValueError("Content cannot be decoded as JSON or YAML: " + str(e)) from e
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
            raise ConnectionError(f"Failed to fetch the specification from URL: {url}. {e!s}") from e
        return cls.from_str(content)

    def find_operation_by_id(self, op_id: str, method: Optional[str] = None) -> Operation:
        """
        Find an operation by operationId.
        """
        for path, path_item in self.spec_dict.get("paths", {}).items():
            op: Operation = self.get_operation_item(path, path_item, method)
            if op_id in op.operation_dict.get("operationId", ""):
                return self.get_operation_item(path, path_item, method)
        raise ValueError(f"No operation found with operationId {op_id}")

    def get_operation_item(self, path: str, path_item: Dict[str, Any], method: Optional[str] = None) -> Operation:
        """
        Get an operation item from the OpenAPI specification.

        :param path: The path of the operation.
        :param path_item: The path item from the OpenAPI specification.
        :param method: The HTTP method of the operation.
        """
        if method:
            operation_dict = path_item.get(method.lower(), {})
            if not operation_dict:
                raise ValueError(f"No operation found for method {method} at path {path}")
            return Operation(path, method.lower(), operation_dict, self.spec_dict)
        if len(path_item) == 1:
            method, operation_dict = next(iter(path_item.items()))
            return Operation(path, method, operation_dict, self.spec_dict)
        if len(path_item) > 1:
            raise ValueError(f"Multiple operations found at path {path}, method parameter is required.")
        raise ValueError(f"No operations found at path {path}.")

    def get_security_schemes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the security schemes from the OpenAPI specification.
        """
        components = self.spec_dict.get("components", {})
        return components.get("securitySchemes", {})

    def to_dict(self, *, resolve_references: Optional[bool] = False) -> Dict[str, Any]:
        """
        Converts the OpenAPI specification to a dictionary format.

        Optionally resolves all $ref references within the spec, returning a fully resolved specification
        dictionary if `resolve_references` is set to True.

        :param resolve_references: If True, resolve references in the specification.
        :return: A dictionary representation of the OpenAPI specification, optionally fully resolved.
        """
        return jsonref.replace_refs(self.spec_dict, proxies=False) if resolve_references else self.spec_dict


class ClientConfiguration:
    """Configuration for the OpenAPI client."""

    def __init__(  # noqa: PLR0913 pylint: disable=too-many-arguments
        self,
        openapi_spec: Union[str, Path, Dict[str, Any]],
        credentials: Optional[Union[str, Dict[str, Any], AuthenticationStrategy]] = None,
        http_client: Optional[HttpClient] = None,
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
            raise ValueError("Invalid OpenAPI specification format. Expected file path or dictionary.")

        self.credentials = credentials
        self.http_client = http_client or HttpClient(HttpClientConfig())
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
            return self._create_authentication_from_string(self.credentials, security_schemes)
        if isinstance(self.credentials, dict):
            return self._create_authentication_from_dict(self.credentials)
        raise ValueError(f"Unsupported credentials type: {type(self.credentials)}")

    def get_tools_definitions(self) -> List[Dict[str, Any]]:
        """
        Get the tools definitions used as tools LLM parameter.
        """
        provider_to_converter = {"anthropic": anthropic_converter, "cohere": cohere_converter}
        converter = provider_to_converter.get(self.llm_provider, openai_converter)
        return converter(self.openapi_spec)

    def get_payload_extractor(self):
        """
        Get the payload extractor for the LLM provider.
        """
        provider_to_arguments_field_name = {"anthropic": "input", "cohere": "parameters"}  # add more providers here
        # default to OpenAI "arguments"
        arguments_field_name = provider_to_arguments_field_name.get(self.llm_provider, "arguments")
        return LLMFunctionPayloadExtractor(arguments_field_name=arguments_field_name)

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
        raise ValueError(f"Unable to create authentication from provided credentials: {credentials}")

    def _create_authentication_from_dict(self, credentials: Dict[str, Any]) -> AuthenticationStrategy:
        if "username" in credentials and "password" in credentials:
            return HTTPAuthentication(username=credentials["username"], password=credentials["password"])
        if "api_key" in credentials:
            return ApiKeyAuthentication(api_key=credentials["api_key"])
        if "token" in credentials:
            return HTTPAuthentication(token=credentials["token"])
        if "access_token" in credentials:
            raise NotImplementedError("OAuth2 authentication is not yet supported.")
        raise ValueError("Unable to create authentication from provided credentials: {credentials}")

    def is_valid_http_url(self, url: str) -> bool:
        """Check if a URL is a valid HTTP/HTTPS URL."""
        r = urlparse(url)
        return all([r.scheme in ["http", "https"], r.netloc])


class LLMFunctionPayloadExtractor:
    """
    Implements a recursive search for extracting LLM generated function payloads.
    """

    def __init__(self, arguments_field_name: str):
        self.arguments_field_name = arguments_field_name

    def extract_function_invocation(self, payload: Any) -> Dict[str, Any]:
        """
        Extract the function invocation details from the payload.
        """
        fields_and_values = self._search(payload)
        if fields_and_values:
            arguments = fields_and_values.get(self.arguments_field_name)
            if not isinstance(arguments, (str, dict)):
                raise ValueError(
                    f"Invalid {self.arguments_field_name} type {type(arguments)} for function call, expected str/dict"
                )
            return {
                "name": fields_and_values.get("name"),
                "arguments": json.loads(arguments) if isinstance(arguments, str) else arguments,
            }
        return {}

    def _required_fields(self) -> List[str]:
        return ["name", self.arguments_field_name]

    def _search(self, payload: Any) -> Dict[str, Any]:
        if self._is_primitive(payload):
            return {}
        if dict_converter := self._get_dict_converter(payload):
            payload = dict_converter()
        elif dataclasses.is_dataclass(payload):
            payload = dataclasses.asdict(payload)
        if isinstance(payload, dict):
            if all(field in payload for field in self._required_fields()):
                # this is the payload we are looking for
                return payload
            for value in payload.values():
                result = self._search(value)
                if result:
                    return result
        elif isinstance(payload, list):
            for item in payload:
                result = self._search(item)
                if result:
                    return result
        return {}

    def _get_dict_converter(
        self, obj: Any, method_names: Optional[List[str]] = None
    ) -> Union[Callable[[], Dict[str, Any]], None]:
        method_names = method_names or ["model_dump", "dict"]  # search for pydantic v2 then v1
        for attr in method_names:
            if hasattr(obj, attr) and callable(getattr(obj, attr)):
                return getattr(obj, attr)
        return None

    def _is_primitive(self, obj) -> bool:
        return isinstance(obj, (int, float, str, bool, type(None)))


class OpenAPIServiceClient:
    """
    A client for invoking operations on REST services defined by OpenAPI specifications.

    Together with the `ClientConfiguration`, its `ClientConfigurationBuilder`, the `OpenAPIServiceClient`
    simplifies the process of (LLMs) with services defined by OpenAPI specifications.
    """

    def __init__(self, client_config: ClientConfiguration):
        self.auth_config = client_config.get_auth_config()
        self.openapi_spec = client_config.openapi_spec
        self.http_client = client_config.http_client
        self.payload_extractor = client_config.get_payload_extractor()

    def invoke(self, function_payload: Any) -> Any:
        """
        Invokes a function specified in the function payload.

        :param function_payload: The function payload containing the details of the function to be invoked.
        :returns: The response from the service after invoking the function.
        :raises OpenAPIClientError: If the function invocation payload cannot be extracted from the function payload.
        :raises HttpClientError: If an error occurs while sending the request and receiving the response.
        """
        fn_invocation_payload = self.payload_extractor.extract_function_invocation(function_payload)
        if not fn_invocation_payload:
            raise OpenAPIClientError(
                f"Failed to extract function invocation payload from {function_payload} using "
                f"{self.payload_extractor.__class__.__name__}. Ensure the payload format matches the expected "
                "structure for the designated LLM extractor."
            )
        # fn_invocation_payload, if not empty, guaranteed to have "name" and "arguments" keys from here on
        operation = self.openapi_spec.find_operation_by_id(fn_invocation_payload.get("name"))
        request = self._build_request(operation, **fn_invocation_payload.get("arguments"))
        self._apply_authentication(self.auth_config, operation, request)
        return self.http_client.send_request(request)

    def _build_request(self, operation: Operation, **kwargs) -> Any:
        request = {
            "url": self._build_url(operation, **kwargs),
            "method": operation.method.lower(),
            "headers": self._build_headers(operation, **kwargs),
            "params": self._build_query_params(operation, **kwargs),
            "json": self._build_request_body(operation, **kwargs),
        }
        return request

    def _build_headers(self, operation: Operation, **kwargs) -> Dict[str, str]:
        headers = {}
        for parameter in operation.get_parameters("header"):
            param_value = kwargs.get(parameter["name"], None)
            if param_value:
                headers[parameter["name"]] = str(param_value)
            elif parameter.get("required", False):
                raise ValueError(f"Missing required header parameter: {parameter['name']}")
        return headers

    def _build_url(self, operation: Operation, **kwargs) -> str:
        server_url = operation.get_server()
        path = operation.path
        for parameter in operation.get_parameters("path"):
            param_value = kwargs.get(parameter["name"], None)
            if param_value:
                path = path.replace(f"{{{parameter['name']}}}", str(param_value))
            elif parameter.get("required", False):
                raise ValueError(f"Missing required path parameter: {parameter['name']}")
        return server_url + path

    def _build_query_params(self, operation: Operation, **kwargs) -> Dict[str, Any]:
        query_params = {}
        for parameter in operation.get_parameters("query"):
            param_value = kwargs.get(parameter["name"], None)
            if param_value:
                query_params[parameter["name"]] = param_value
            elif parameter.get("required", False):
                raise ValueError(f"Missing required query parameter: {parameter['name']}")
        return query_params

    def _build_request_body(self, operation: Operation, **kwargs) -> Any:
        request_body = operation.request_body
        if request_body:
            content = request_body.get("content", {})
            if "application/json" in content:
                return {**kwargs}
            raise NotImplementedError("Request body content type not supported")
        return None

    def _apply_authentication(self, auth: AuthenticationStrategy, operation: Operation, request: Dict[str, Any]):
        auth_config = auth or AuthenticationStrategy()
        security_requirements = operation.security_requirements
        security_schemes = operation.spec_dict.get("components", {}).get("securitySchemes", {})
        if security_requirements:
            for requirement in security_requirements:
                for scheme_name in requirement:
                    if scheme_name in security_schemes:
                        security_scheme = security_schemes[scheme_name]
                        auth_config.apply_auth(security_scheme, request)
                    break


class OpenAPIClientError(Exception):
    """Exception raised for errors in the OpenAPI client."""


def openai_converter(schema: OpenAPISpecification) -> List[Dict[str, Any]]:
    """
    Converts OpenAPI specification to a list of function suitable for OpenAI LLM function calling.

    :param schema: The OpenAPI specification to convert.
    :return: A list of dictionaries, each representing a function definition.
    """
    resolved_schema = jsonref.replace_refs(schema.spec_dict)
    fn_definitions = _openapi_to_functions(resolved_schema, "parameters", _parse_endpoint_spec_openai)
    return [{"type": "function", "function": fn} for fn in fn_definitions]


def anthropic_converter(schema: OpenAPISpecification) -> List[Dict[str, Any]]:
    """
    Converts an OpenAPI specification to a list of function definitions for Anthropic LLM function calling.

    :param schema: The OpenAPI specification to convert.
    :return: A list of dictionaries, each representing a function definition.
    """
    resolved_schema = jsonref.replace_refs(schema.spec_dict)
    return _openapi_to_functions(resolved_schema, "input_schema", _parse_endpoint_spec_openai)


def cohere_converter(schema: OpenAPISpecification) -> List[Dict[str, Any]]:
    """
    Converts an OpenAPI specification to a list of function definitions for Cohere LLM function calling.

    :param schema: The OpenAPI specification to convert.
    :return: A list of dictionaries, each representing a function definition.
    """
    resolved_schema = jsonref.replace_refs(schema.spec_dict)
    return _openapi_to_functions(resolved_schema, "not important for cohere", _parse_endpoint_spec_cohere)


def _openapi_to_functions(
    service_openapi_spec: Dict[str, Any],
    parameters_name: str,
    parse_endpoint_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extracts functions from the OpenAPI specification, converts them into a function schema.
    """

    # Doesn't enforce rigid spec validation because that would require a lot of dependencies
    # We check the version and require minimal fields to be present, so we can extract functions
    spec_version = service_openapi_spec.get("openapi")
    if not spec_version:
        raise ValueError(f"Invalid OpenAPI spec provided. Could not extract version from {service_openapi_spec}")
    service_openapi_spec_version = int(spec_version.split(".")[0])
    # Compare the versions
    if service_openapi_spec_version < MIN_REQUIRED_OPENAPI_SPEC_VERSION:
        raise ValueError(
            f"Invalid OpenAPI spec version {service_openapi_spec_version}. Must be "
            f"at least {MIN_REQUIRED_OPENAPI_SPEC_VERSION}."
        )
    functions: List[Dict[str, Any]] = []
    for paths in service_openapi_spec["paths"].values():
        for path_spec in paths.values():
            function_dict = parse_endpoint_fn(path_spec, parameters_name)
            if function_dict:
                functions.append(function_dict)
    return functions


def _parse_endpoint_spec_openai(resolved_spec: Dict[str, Any], parameters_name: str) -> Dict[str, Any]:
    """
    Parses an OpenAPI endpoint specification for OpenAI.
    """
    if not isinstance(resolved_spec, dict):
        logger.warning("Invalid OpenAPI spec format provided. Could not extract function.")
        return {}
    function_name = resolved_spec.get("operationId")
    description = resolved_spec.get("description") or resolved_spec.get("summary", "")
    schema: Dict[str, Any] = {"type": "object", "properties": {}}
    # requestBody section
    req_body_schema = (
        resolved_spec.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
    )
    if "properties" in req_body_schema:
        for prop_name, prop_schema in req_body_schema["properties"].items():
            schema["properties"][prop_name] = _parse_property_attributes(prop_schema)
        if "required" in req_body_schema:
            schema.setdefault("required", []).extend(req_body_schema["required"])

    # parameters section
    for param in resolved_spec.get("parameters", []):
        if "schema" in param:
            schema_dict = _parse_property_attributes(param["schema"])
            # these attributes are not in param[schema] level but on param level
            useful_attributes = ["description", "pattern", "enum"]
            schema_dict.update({key: param[key] for key in useful_attributes if param.get(key)})
            schema["properties"][param["name"]] = schema_dict
            if param.get("required", False):
                schema.setdefault("required", []).append(param["name"])

    if function_name and description and schema["properties"]:
        return {"name": function_name, "description": description, parameters_name: schema}
    logger.warning("Invalid OpenAPI spec format provided. Could not extract function from %s", resolved_spec)
    return {}


def _parse_property_attributes(
    property_schema: Dict[str, Any], include_attributes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Recursively parses the attributes of a property schema.
    """
    include_attributes = include_attributes or ["description", "pattern", "enum"]
    schema_type = property_schema.get("type")
    parsed_schema = {"type": schema_type} if schema_type else {}
    for attr in include_attributes:
        if attr in property_schema:
            parsed_schema[attr] = property_schema[attr]
    if schema_type == "object":
        properties = property_schema.get("properties", {})
        parsed_properties = {
            prop_name: _parse_property_attributes(prop, include_attributes) for prop_name, prop in properties.items()
        }
        parsed_schema["properties"] = parsed_properties
        if "required" in property_schema:
            parsed_schema["required"] = property_schema["required"]
    elif schema_type == "array":
        items = property_schema.get("items", {})
        parsed_schema["items"] = _parse_property_attributes(items, include_attributes)
    return parsed_schema


def _parse_endpoint_spec_cohere(operation: Dict[str, Any], ignored_param: str) -> Dict[str, Any]:
    """
    Parses an endpoint specification for Cohere.
    """
    function_name = operation.get("operationId")
    description = operation.get("description") or operation.get("summary", "")
    parameter_definitions = _parse_parameters(operation)
    if function_name:
        return {
            "name": function_name,
            "description": description,
            "parameter_definitions": parameter_definitions,
        }
    logger.warning("Operation missing operationId, cannot create function definition.")
    return {}


def _parse_parameters(operation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses the parameters from an operation specification.
    """
    parameters = {}
    for param in operation.get("parameters", []):
        if "schema" in param:
            parameters[param["name"]] = _parse_schema(
                param["schema"], param.get("required", False), param.get("description", "")
            )
    if "requestBody" in operation:
        content = operation["requestBody"].get("content", {}).get("application/json", {})
        if "schema" in content:
            schema_properties = content["schema"].get("properties", {})
            required_properties = content["schema"].get("required", [])
            for name, schema in schema_properties.items():
                parameters[name] = _parse_schema(schema, name in required_properties, schema.get("description", ""))
    return parameters


def _parse_schema(schema: Dict[str, Any], required: bool, description: str) -> Dict[str, Any]:  # noqa: FBT001
    """
    Parses a schema part of an operation specification.
    """
    schema_type = _get_type(schema)
    if schema_type == "object":
        # Recursive call for complex types
        properties = schema.get("properties", {})
        nested_parameters = {
            name: _parse_schema(
                schema=prop_schema,
                required=bool(name in schema.get("required", False)),
                description=prop_schema.get("description", ""),
            )
            for name, prop_schema in properties.items()
        }
        return {
            "type": schema_type,
            "description": description,
            "properties": nested_parameters,
            "required": required,
        }
    return {"type": schema_type, "description": description, "required": required}


def _get_type(schema: Dict[str, Any]) -> str:
    type_mapping = {
        "integer": "int",
        "string": "str",
        "boolean": "bool",
        "number": "float",
        "object": "object",
        "array": "list",
    }
    schema_type = schema.get("type", "object")
    if schema_type not in type_mapping:
        raise ValueError(f"Unsupported schema type {schema_type}")
    return type_mapping[schema_type]
