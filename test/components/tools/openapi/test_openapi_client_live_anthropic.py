# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import anthropic
import pytest

from haystack_experimental.components.tools.openapi.openapi import ClientConfiguration, OpenAPIServiceClient


class TestClientLiveAnthropic:

    @pytest.mark.skipif("SERPERDEV_API_KEY" not in os.environ, reason="SERPERDEV_API_KEY not set")
    @pytest.mark.skipif("ANTHROPIC_API_KEY" not in os.environ, reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    def test_serperdev(self, test_files_path):
        config = ClientConfiguration(openapi_spec=test_files_path / "yaml" / "serper.yml",
                                     credentials=os.getenv("SERPERDEV_API_KEY"),
                                     llm_provider="anthropic")
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.beta.tools.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            tools=config.get_tools_definitions(),
            messages=[{"role": "user", "content": "Do a google search: Who was Nikola Tesla?"}],
        )
        service_api = OpenAPIServiceClient(config)
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

        # make a few more requests to test the same tool
        service_response = service_api.invoke(response)
        assert "Serbian" in str(service_response)

        service_response = service_api.invoke(response)
        assert "American" in str(service_response)

    @pytest.mark.skipif("ANTHROPIC_API_KEY" not in os.environ, reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    def test_github(self, test_files_path):
        config = ClientConfiguration(openapi_spec=test_files_path / "yaml" / "github_compare.yml",
                                     llm_provider="anthropic")

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.beta.tools.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            tools=config.get_tools_definitions(),
            messages=[
                {
                    "role": "user",
                    "content": "Compare branches main and add_default_adapter_filters in repo"
                    " haystack and owner deepset-ai",
                }
            ],
        )
        service_api = OpenAPIServiceClient(config)
        service_response = service_api.invoke(response)
        assert "deepset" in str(service_response)
