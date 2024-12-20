# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Tool class OpenAPI functionality."""

import os
import pytest

from haystack_experimental.tools.openapi import OpenAPIKwargs, _create_tools_from_openapi_spec
from haystack_experimental.dataclasses.tool import Tool


class TestToolOpenAPI:
    """Test suite for Tool's OpenAPI integration capabilities."""

    @pytest.mark.skipif(
        not os.getenv("SERPERDEV_API_KEY"),
        reason="SERPERDEV_API_KEY environment variable not set"
    )
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set"
    )
    def test_from_openapi_spec_serperdev(self):
        """
        Test creating a Tool from SerperDev's OpenAPI specification.

        Verifies that a Tool can be properly created from SerperDev's OpenAPI spec using
        all supported ways of passing configuration.
        """

        serper_api_key = os.getenv("SERPERDEV_API_KEY")
        assert serper_api_key is not None

        # Direct kwargs usage
        tools = _create_tools_from_openapi_spec(
            spec="https://bit.ly/serperdev_openapi",
            credentials=serper_api_key
        )
        assert len(tools) >= 1
        tool = tools[0]
        assert tool.name == "search"
        assert "search" in tool.description.lower()

        # Constructor-style creation and unpacking
        config = OpenAPIKwargs(
            credentials=serper_api_key
        )
        tools = _create_tools_from_openapi_spec(spec="https://bit.ly/serperdev_openapi", **config)
        assert len(tools) >= 1
        tool = tools[0]
        assert tool.name == "search"
        assert "search" in tool.description.lower()

        # Dict-style creation and unpacking
        config = OpenAPIKwargs(**{
            "credentials": serper_api_key
        })
        tools = _create_tools_from_openapi_spec(spec="https://bit.ly/serperdev_openapi", **config)
        assert len(tools) >= 1
        tool = tools[0]
        assert tool.name == "search"
        assert "search" in tool.description.lower()

    @pytest.mark.skipif(
        not os.getenv("SERPERDEV_API_KEY"),
        reason="SERPERDEV_API_KEY environment variable not set"
    )
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set"
    )
    def test_from_openapi_spec_serperdev_with_allowed_operations(self):
        """
        Test creating a Tool from SerperDev's OpenAPI specification
        with allowed operations.

        Verifies that a Tool can be properly created from SerperDev's OpenAPI spec using
        all supported ways of passing configuration and that only the allowed operations are used.
        """
        serper_api_key = os.getenv("SERPERDEV_API_KEY")
        assert serper_api_key is not None

        # Direct kwargs usage
        tools = _create_tools_from_openapi_spec(
            spec="https://bit.ly/serperdev_openapi",
            credentials=serper_api_key,
            allowed_operations=["search"]
        )
        assert len(tools) >= 1
        tool = tools[0]
        assert tool.name == "search"
        assert "search" in tool.description.lower()

        # Constructor-style creation and unpacking
        config = OpenAPIKwargs(
            credentials=serper_api_key,
            allowed_operations=["search"]
        )
        tools = _create_tools_from_openapi_spec(spec="https://bit.ly/serperdev_openapi", **config)
        assert len(tools) >= 1
        tool = tools[0]
        assert tool.name == "search"
        assert "search" in tool.description.lower()

        # Dict-style creation and unpacking
        config = OpenAPIKwargs(**{
            "credentials": serper_api_key,
            "allowed_operations": ["search"]
        })
        tools = _create_tools_from_openapi_spec(spec="https://bit.ly/serperdev_openapi", **config)
        assert len(tools) >= 1
        tool = tools[0]
        assert tool.name == "search"
        assert "search" in tool.description.lower()

        # Test with non-existent operation
        config = OpenAPIKwargs(
            credentials=serper_api_key,
            allowed_operations=["super-search"]
        )
        tools = _create_tools_from_openapi_spec(spec="https://bit.ly/serperdev_openapi", **config)
        assert len(tools) == 0

    @pytest.mark.skipif(
        not os.getenv("SERPERDEV_API_KEY"),
        reason="SERPERDEV_API_KEY environment variable not set"
    )
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set"
    )
    def test_tool_from_openapi_serperdev(self):
        """
        Test creating a Tool using Tool.from_openapi with SerperDev's OpenAPI specification.

        Verifies that a Tool can be properly created using the Tool.from_openapi method with
        all supported ways of passing configuration.
        """
        serper_api_key = os.getenv("SERPERDEV_API_KEY")
        assert serper_api_key is not None

        # Direct kwargs usage
        tool = Tool.from_openapi(
            spec="https://bit.ly/serperdev_openapi",
            operation_name="search",
            credentials=serper_api_key
        )
        assert tool.name == "search"
        assert "search" in tool.description.lower()

        # Constructor-style creation and unpacking
        config = OpenAPIKwargs(
            credentials=serper_api_key
        )
        tool = Tool.from_openapi(
            spec="https://bit.ly/serperdev_openapi",
            operation_name="search",
            **config
        )
        assert tool.name == "search"
        assert "search" in tool.description.lower()

        # Dict-style creation and unpacking
        config = OpenAPIKwargs(**{
            "credentials": serper_api_key
        })
        tool = Tool.from_openapi(
            spec="https://bit.ly/serperdev_openapi",
            operation_name="search",
            **config
        )
        assert tool.name == "search"
        assert "search" in tool.description.lower()

    @pytest.mark.skipif(
        not os.getenv("SERPERDEV_API_KEY"),
        reason="SERPERDEV_API_KEY environment variable not set"
    )
    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY environment variable not set"
    )
    def test_tool_from_openapi_serperdev_invalid_operation(self):
        """
        Test Tool.from_openapi with an invalid operation name.

        Verifies that attempting to create a Tool with a non-existent operation
        raises an appropriate error.
        """
        serper_api_key = os.getenv("SERPERDEV_API_KEY")
        assert serper_api_key is not None

        with pytest.raises(ValueError):
            Tool.from_openapi(
                spec="https://bit.ly/serperdev_openapi",
                operation_name="super-search",
                credentials=serper_api_key
            )
