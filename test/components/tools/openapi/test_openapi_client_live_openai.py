# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from openai import OpenAI

from haystack_experimental.components.tools.openapi._openapi import ClientConfiguration, OpenAPIServiceClient
from test.components.tools.openapi.conftest import create_openapi_spec


class TestClientLiveOpenAPI:

    @pytest.mark.skipif("SERPERDEV_API_KEY" not in os.environ, reason="SERPERDEV_API_KEY not set")
    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_serperdev(self, test_files_path):

        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "serper.yml"),
                                     credentials=os.getenv("SERPERDEV_API_KEY"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Do a serperdev google search: Who was Nikola Tesla?"}],
            tools=config.get_tools_definitions(),
        )
        service_api = OpenAPIServiceClient(config)
        service_response = service_api.invoke(response)
        assert "inventions" in str(service_response)

        # make a few more requests to test the same tool
        service_response = service_api.invoke(response)
        assert "Serbian" in str(service_response)

        service_response = service_api.invoke(response)
        assert "American" in str(service_response)

    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    @pytest.mark.skip("This test hits rate limit on Github API. Skip for now.")
    def test_github(self, test_files_path):
        config = ClientConfiguration(openapi_spec=create_openapi_spec(test_files_path / "yaml" / "github_compare.yml"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": "Compare branches main and add_default_adapter_filters in repo"
                    " haystack and owner deepset-ai",
                }
            ],
            tools=config.get_tools_definitions(),
        )
        service_api = OpenAPIServiceClient(config)
        service_response = service_api.invoke(response)
        assert "deepset" in str(service_response)

    @pytest.mark.skipif("FIRECRAWL_API_KEY" not in os.environ, reason="FIRECRAWL_API_KEY not set")
    @pytest.mark.skipif("OPENAI_API_KEY" not in os.environ, reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    @pytest.mark.skip("This test is flaky likely due to load on the popular Firecrawl API. Skip for now.")
    def test_firecrawl(self):
        openapi_spec_url = "https://raw.githubusercontent.com/mendableai/firecrawl/main/apps/api/openapi.json"
        config = ClientConfiguration(openapi_spec=create_openapi_spec(openapi_spec_url), credentials=os.getenv("FIRECRAWL_API_KEY"))
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Scrape URL: https://news.ycombinator.com/"}],
            tools=config.get_tools_definitions(),
        )
        service_api = OpenAPIServiceClient(config)
        service_response = service_api.invoke(response)
        assert isinstance(service_response, dict)
        assert service_response.get("success", False), "Firecrawl scrape API call failed"

        # now test the same openapi service but different endpoint/tool
        top_k = 2
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": f"Search Google for `Why was Sam Altman ousted from OpenAI?`, limit to {top_k} results",
                }
            ],
            tools=config.get_tools_definitions(),
        )
        service_response = service_api.invoke(response)
        assert isinstance(service_response, dict)
        assert service_response.get("success", False), "Firecrawl search API call failed"
        assert len(service_response.get("data", [])) == top_k
        assert "Sam" in str(service_response)
