# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os

import cohere
import pytest

from haystack_experimental.util.openapi import ClientConfigurationBuilder, OpenAPIServiceClient

# Copied from Cohere's documentation
preamble = """
## Task & Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of
 requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to
 help you, which you use to research your answer. You should focus on serving the user's needs as best you can,
 which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and
 spelling.
"""


class TestClientLiveCohere:

    @pytest.mark.skipif("SERPERDEV_API_KEY" not in os.environ, reason="SERPERDEV_API_KEY not set")
    @pytest.mark.skipif("COHERE_API_KEY" not in os.environ, reason="COHERE_API_KEY not set")
    @pytest.mark.integration
    def test_serperdev(self, test_files_path):
        builder = ClientConfigurationBuilder()
        config = (
            builder.with_openapi_spec(test_files_path / "yaml" / "serper.yml")
            .with_credentials(os.getenv("SERPERDEV_API_KEY"))
            .with_provider("cohere")
            .build()
        )
        client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        response = client.chat(
            model="command-r",
            preamble=preamble,
            tools=config.get_tools_definitions(),
            message="Do a google search: Who was Nikola Tesla?",
        )
        service_api = OpenAPIServiceClient(config)
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

        # make a few more requests to test the same tool
        service_response = service_api.invoke(response)
        assert "Serbian" in str(service_response)

        service_response = service_api.invoke(response)
        assert "American" in str(service_response)

    @pytest.mark.skipif("COHERE_API_KEY" not in os.environ, reason="COHERE_API_KEY not set")
    @pytest.mark.integration
    def test_github(self, test_files_path):
        builder = ClientConfigurationBuilder()
        config = (
            builder.with_openapi_spec(test_files_path / "yaml" / "github_compare.yml").with_provider("cohere").build()
        )

        client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        response = client.chat(
            model="command-r",
            preamble=preamble,
            tools=config.get_tools_definitions(),
            message="Compare branches main and add_default_adapter_filters in repo haystack and owner deepset-ai",
        )
        service_api = OpenAPIServiceClient(config)
        service_response = service_api.invoke(response)
        assert "deepset" in str(service_response)
