# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.url_validation import is_valid_http_url

from haystack_experimental.components.tools.openapi._openapi import (
    ClientConfiguration,
    OpenAPIServiceClient,
)
from haystack_experimental.components.tools.openapi.types import LLMProvider, OpenAPISpecification
from haystack_experimental.util import serialize_secrets_inplace

with LazyImport("Run 'pip install anthropic-haystack'") as anthropic_import:
    # pylint: disable=import-error
    from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator

with LazyImport("Run 'pip install cohere-haystack'") as cohere_import:
    # pylint: disable=import-error
    from haystack_integrations.components.generators.cohere import CohereChatGenerator

logger = logging.getLogger(__name__)


@component
class OpenAPITool:
    """
    The OpenAPITool calls a RESTful endpoint of an OpenAPI service using payloads generated from human instructions.

    Here is an example of how to use the OpenAPITool component to scrape a URL using the FireCrawl API:

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_experimental.components.tools.openapi import OpenAPITool, LLMProvider
    from haystack.utils import Secret

    tool = OpenAPITool(generator_api=LLMProvider.OPENAI,
                       generator_api_params={"model":"gpt-4o-mini"},
                       spec="https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json",
                       credentials=Secret.from_env_var("FIRECRAWL_API_KEY"))

    results = tool.run(messages=[ChatMessage.from_user("Scrape URL: https://news.ycombinator.com/")])
    print(results)
    ```

    Similarly, you can use the OpenAPITool component to invoke **any** OpenAPI service/tool by providing the OpenAPI
    specification and credentials.
    """

    def __init__(
        self,
        generator_api: LLMProvider,
        generator_api_params: Optional[Dict[str, Any]] = None,
        spec: Optional[Union[str, Path]] = None,
        credentials: Optional[Secret] = None,
        allowed_operations: Optional[List[str]] = None,
    ):  # pylint: disable=too-many-positional-arguments
        """
        Initialize the OpenAPITool component.

        :param generator_api: The API provider for the chat generator.
        :param generator_api_params: Parameters to pass for the chat generator creation.
        :param spec: OpenAPI specification for the tool/service. This can be a URL, a local file path, or
        an OpenAPI service specification provided as a string.
        :param credentials: Credentials for the tool/service.
        :param allowed_operations: A list of operations to register with LLMs via the LLM tools parameter. Use
        operationId field in the OpenAPI spec path/operation to specify the operation names to use. If not specified,
        all operations found in the OpenAPI spec will be registered with LLMs.
        """
        self.generator_api = generator_api
        self.generator_api_params = generator_api_params or {}  # store the generator API parameters for serialization
        self.chat_generator = self._init_generator(generator_api, generator_api_params or {})
        self.config_openapi: Optional[ClientConfiguration] = None
        self.open_api_service: Optional[OpenAPIServiceClient] = None
        self.spec = spec  # store the spec for serialization
        self.credentials = credentials  # store the credentials for serialization
        self.allowed_operations = allowed_operations
        if spec:
            if os.path.isfile(spec):
                openapi_spec = OpenAPISpecification.from_file(spec)
            elif is_valid_http_url(str(spec)):
                openapi_spec = OpenAPISpecification.from_url(str(spec))
            else:
                raise ValueError(f"Invalid OpenAPI specification source {spec}. Expected valid file path or URL")
            self.config_openapi = ClientConfiguration(
                openapi_spec=openapi_spec,
                credentials=credentials.resolve_value() if credentials else None,
                llm_provider=generator_api,
                operations_filter=(lambda f: f["operationId"] in allowed_operations) if allowed_operations else None,
            )
            self.open_api_service = OpenAPIServiceClient(self.config_openapi)

    @component.output_types(service_response=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        fc_generator_kwargs: Optional[Dict[str, Any]] = None,
        spec: Optional[Union[str, Path]] = None,
        credentials: Optional[Secret] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Invokes the underlying OpenAPI service/tool with the function calling payload generated by the chat generator.

        :param messages: List of ChatMessages to generate function calling payload (e.g. human instructions). The last
        message should be human instruction containing enough information to generate the function calling payload
        suitable for the OpenAPI service/tool used. See the examples in the class docstring.
        :param fc_generator_kwargs: Additional arguments for the function calling payload generation process.
        :param spec: OpenAPI specification for the tool/service, overrides the one provided at initialization.
        :param credentials: Credentials for the tool/service, overrides the one provided at initialization.
        :returns: a dictionary containing the service response with the following key:
            - `service_response`: List of ChatMessages containing the service response. ChatMessages are generated
            based on the response from the OpenAPI service/tool and contains the JSON response from the service.
            If there is an error during the invocation, the response will be a ChatMessage with the error message under
            the `error` key.
        """
        last_message = messages[-1]
        if not last_message.is_from(ChatRole.USER):
            raise ValueError(f"{last_message} not from the user")
        if not last_message.content:
            raise ValueError("Function calling instruction message content is empty.")

        # build a new ClientConfiguration and OpenAPIServiceClient if a runtime tool_spec is provided
        openapi_service: Optional[OpenAPIServiceClient] = self.open_api_service
        config_openapi: Optional[ClientConfiguration] = self.config_openapi
        if spec:
            if os.path.isfile(spec):
                openapi_spec = OpenAPISpecification.from_file(spec)
            elif is_valid_http_url(str(spec)):
                openapi_spec = OpenAPISpecification.from_url(str(spec))
            else:
                raise ValueError(f"Invalid OpenAPI specification source {spec}. Expected valid file path or URL")

            config_openapi = ClientConfiguration(
                openapi_spec=openapi_spec,
                credentials=credentials.resolve_value() if credentials else None,
                llm_provider=self.generator_api,
            )
            openapi_service = OpenAPIServiceClient(config_openapi)

        if not openapi_service or not config_openapi:
            raise ValueError(
                "OpenAPI specification not provided. Please provide an OpenAPI specification either at initialization "
                "or during runtime."
            )

        # merge fc_generator_kwargs, tools definitions comes from the OpenAPI spec, other kwargs are passed by the user
        fc_generator_kwargs = {
            "tools": config_openapi.get_tools_definitions(),
            **(fc_generator_kwargs or {}),
        }

        # generate function calling payload with the chat generator
        logger.debug(
            "Invoking chat generator with {message} to generate function calling payload.",
            message=last_message.content,
        )
        fc_payload = self.chat_generator.run(messages, generation_kwargs=fc_generator_kwargs)
        try:
            invocation_payload = json.loads(fc_payload["replies"][0].content)
            logger.debug("Invoking tool with {payload}", payload=invocation_payload)
            service_response = openapi_service.invoke(invocation_payload)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error invoking OpenAPI endpoint. Error: {e}", e=str(e))
            service_response = {"error": str(e)}
        response_messages = [ChatMessage.from_user(json.dumps(service_response))]

        return {"service_response": response_messages}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        serialize_secrets_inplace(self.generator_api_params, keys=["api_key"], recursive=True)
        return default_to_dict(
            self,
            generator_api=self.generator_api.value,
            generator_api_params=self.generator_api_params,
            spec=self.spec,
            credentials=self.credentials.to_dict() if self.credentials else None,
            allowed_operations=self.allowed_operations,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAPITool":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["credentials"])
        deserialize_secrets_inplace(data["init_parameters"]["generator_api_params"], keys=["api_key"])
        init_params = data.get("init_parameters", {})
        generator_api = init_params.get("generator_api")
        data["init_parameters"]["generator_api"] = LLMProvider.from_str(generator_api)
        return default_from_dict(cls, data)

    def _init_generator(self, generator_api: LLMProvider, generator_api_params: Dict[str, Any]):
        """
        Initialize the chat generator based on the specified API provider and parameters.
        """
        if generator_api == LLMProvider.OPENAI:
            return OpenAIChatGenerator(**generator_api_params)
        if generator_api == LLMProvider.COHERE:
            cohere_import.check()
            return CohereChatGenerator(**generator_api_params)
        if generator_api == LLMProvider.ANTHROPIC:
            anthropic_import.check()
            return AnthropicChatGenerator(**generator_api_params)
        raise ValueError(f"Unsupported generator API: {generator_api}")
