# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import component, logging
from haystack.dataclasses import ChatMessage, ChatRole

from haystack_experimental.components.tools.openapi.generator_factory import (
    ChatGeneratorDescriptorManager,
)
from haystack_experimental.components.tools.openapi.openapi import (
    ClientConfiguration,
    OpenAPIServiceClient,
)

logger = logging.getLogger(__name__)


@component
class OpenAPITool:
    """
    The OpenAPITool calls an OpenAPI service using payloads generated by the chat generator from human instructions.

    Here is an example of how to use the OpenAPITool component to scrape a URL using the FireCrawl API:

    ```python
    from haystack_experimental.components.tools import OpenAPITool
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    tool = OpenAPITool(model="gpt-3.5-turbo",
                       tool_spec="https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json",
                       tool_credentials="<your-tool-token>")

    results = tool.run(messages=[ChatMessage.from_user("Scrape URL: https://news.ycombinator.com/")])
    print(results)
    ```

    Similarly, you can use the OpenAPITool component to invoke **any** OpenAPI service/tool by providing the OpenAPI
    specification and credentials.
    """

    def __init__(
        self,
        model: str,
        tool_spec: Optional[Union[str, Path]] = None,
        tool_credentials: Optional[Union[str, Dict[str, Any]]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OpenAPITool component.

        :param model: Name of the chat generator model to use.
        :param tool_spec: OpenAPI specification for the tool/service.
        :param tool_credentials: Credentials for the tool/service.
        :param model_kwargs: Additional arguments for the chat generator model.
        """
        manager = ChatGeneratorDescriptorManager()
        self.descriptor, self.chat_generator = manager.create_generator(
            model_name=model, **(model_kwargs or {})
        )
        self.config_openapi: Optional[ClientConfiguration] = None
        self.open_api_service: Optional[OpenAPIServiceClient] = None
        if tool_spec:
            self.config_openapi = ClientConfiguration(
                openapi_spec=tool_spec,
                credentials=tool_credentials,
                llm_provider=self.descriptor.name,
            )
            self.open_api_service = OpenAPIServiceClient(self.config_openapi)

    @component.output_types(service_response=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        fc_generator_kwargs: Optional[Dict[str, Any]] = None,
        tool_spec: Optional[Union[str, Path, Dict[str, Any]]] = None,
        tool_credentials: Optional[Union[dict, str]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        """
        Invokes the underlying OpenAPI service/tool with the function calling payload generated by the chat generator.

        :param messages: List of ChatMessages to generate function calling payload (e.g. human instructions).
        :param fc_generator_kwargs: Additional arguments for the function calling payload generation process.
        :param tool_spec: OpenAPI specification for the tool/service, overrides the one provided at initialization.
        :param tool_credentials: Credentials for the tool/service, overrides the one provided at initialization.
        :returns: a dictionary containing the service response with the following key:
            - `service_response`: List of ChatMessages containing the service response.
        """
        last_message = messages[-1]
        if not last_message.is_from(ChatRole.USER):
            raise ValueError(f"{last_message} not from the user")
        if not last_message.content:
            raise ValueError("Function calling instruction message content is empty.")

        # build a new ClientConfiguration if a runtime tool_spec is provided
        config_openapi = (
            ClientConfiguration(
                openapi_spec=tool_spec,
                credentials=tool_credentials,
                llm_provider=self.descriptor.name,
            )
            if tool_spec
            else self.config_openapi
        )

        openapi_service: Optional[OpenAPIServiceClient] = self.open_api_service
        if tool_spec:
            config_openapi = ClientConfiguration(
                openapi_spec=tool_spec,
                credentials=tool_credentials,
                llm_provider=self.descriptor.name,
            )
            openapi_service = OpenAPIServiceClient(config_openapi)
        else:
            config_openapi = self.config_openapi

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
        fc_payload = self.chat_generator.run(messages, fc_generator_kwargs)
        try:
            invocation_payload = json.loads(fc_payload["replies"][0].content)
            logger.debug("Invoking tool with {payload}", payload=invocation_payload)
            service_response = openapi_service.invoke(invocation_payload)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error invoking OpenAPI endpoint. Error: {e}", e=str(e))
            service_response = {"error": str(e)}
        response_messages = [ChatMessage.from_user(json.dumps(service_response))]
        return {"service_response": response_messages}
