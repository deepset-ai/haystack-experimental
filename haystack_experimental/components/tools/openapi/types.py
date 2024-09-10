# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import requests
import yaml
from haystack.lazy_imports import LazyImport

from haystack_experimental.components.tools.utils import normalize_function_name

with LazyImport("Run 'pip install jsonref'") as jsonref_import:
    # pylint: disable=import-error
    import jsonref


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


def path_to_operation_id(path: str, http_method: str = "get") -> str:
    """
    Converts an OpenAPI spec path to synthetic operationId.

    :param path: The OpenAPI spec path to convert.
    :param http_method: The HTTP method to use for the operationId.
    :returns: The operationId.
    """
    return normalize_function_name(path + "_" + http_method.lower())


class LLMProvider(Enum):
    """
    LLM providers supported by `OpenAPITool`.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"

    @staticmethod
    def from_str(string: str) -> "LLMProvider":
        """
        Convert a string to a LLMProvider enum.
        """
        provider_map = {e.value: e for e in LLMProvider}
        provider = provider_map.get(string)
        if provider is None:
            msg = (
                f"Invalid LLMProvider '{string}'"
                f"Supported LLMProviders are: {list(provider_map.keys())}"
            )
            raise ValueError(msg)
        return provider


@dataclass
class Operation:
    """
    Represents an operation in an OpenAPI specification

    See https://spec.openapis.org/oas/latest.html#paths-object for details.
    Path objects can contain multiple operations, each with a unique combination of path and method.

    :param path: Path of the operation.
    :param method: HTTP method of the operation.
    :param operation_dict: Operation details from OpenAPI spec
    :param spec_dict: The encompassing OpenAPI specification.
    :param security_requirements: A list of security requirements for the operation.
    :param request_body: Request body details.
    :param parameters: Parameters for the operation.
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
        :returns: The parameters for the operation as a list of dictionaries.
        """
        if location:
            return [param for param in self.parameters if param["in"] == location]
        return self.parameters

    def get_server(self, server_index: int = 0) -> str:
        """
        Get the servers for the operation.

        :param server_index: The index of the server to use.
        :returns: The server URL.
        :raises ValueError: If no servers are found in the specification.
        """
        # servers can be defined at the operation level, path level, or at the root level
        # search for servers in the following order: operation, path, root
        servers = (
            self.operation_dict.get("servers", [])
            or self.spec_dict.get("paths", {}).get(self.path, {}).get("servers", [])
            or self.spec_dict.get("servers", [])
        )
        if not servers:
            raise ValueError("No servers found in the provided specification.")
        if not 0 <= server_index < len(servers):
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
        """
        Initialize an OpenAPISpecification instance.

        :param spec_dict: The OpenAPI specification as a dictionary.
        """
        jsonref_import.check()
        if not isinstance(spec_dict, Dict):
            raise ValueError(
                f"Invalid OpenAPI specification, expected a dictionary: {spec_dict}"
            )
        # just a crude sanity check, by no means a full validation
        if "openapi" not in spec_dict or "paths" not in spec_dict:
            raise ValueError(
                "Invalid OpenAPI specification format. See https://swagger.io/specification/ for details.",
                spec_dict,
            )
        self.spec_dict = jsonref.replace_refs(spec_dict)

    @classmethod
    def from_str(cls, content: str) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a string.

        :param content: The string content of the OpenAPI specification.
        :returns: The OpenAPISpecification instance.
        :raises ValueError: If the content cannot be decoded as JSON or YAML.
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
        :returns: The OpenAPISpecification instance.
        :raises FileNotFoundError: If the specified file does not exist.
        :raises IOError: If an I/O error occurs while reading the file.
        :raises ValueError: If the file content cannot be decoded as JSON or YAML.
        """
        with open(spec_file, encoding="utf-8") as file:
            content = file.read()
        return cls.from_str(content)

    @classmethod
    def from_url(cls, url: str) -> "OpenAPISpecification":
        """
        Create an OpenAPISpecification instance from a URL.

        :param url: The URL to fetch the OpenAPI specification from.
        :returns: The OpenAPISpecification instance.
        :raises ConnectionError: If fetching the specification from the URL fails.
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

    def find_operation_by_id(self, op_id: str) -> Operation:
        """
        Find an Operation by operationId.

        :param op_id: The operationId of the operation.
        :returns: The matching operation
        :raises ValueError: If no operation is found with the given operationId.
        """
        for path, path_value in self.spec_dict.get("paths", {}).items():
            operations = {
                method: operation_dict
                for method, operation_dict in path_value.items()
                if method.lower() in VALID_HTTP_METHODS
            }

            for method, operation_dict in operations.items():
                operation_id = operation_dict.get("operationId", path_to_operation_id(path, method))
                if normalize_function_name(operation_id) == op_id:
                    return Operation(path, method, operation_dict, self.spec_dict)
        raise ValueError(f"No operation found with operationId {op_id}")

    def get_security_schemes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the security schemes from the OpenAPI specification.

        :returns: The security schemes as a dictionary.
        """
        return self.spec_dict.get("components", {}).get("securitySchemes", {})
