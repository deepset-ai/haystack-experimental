# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import patch, MagicMock
from haystack.tools import create_tool_from_function

from haystack_experimental.components.agents.human_in_the_loop.dataclasses import ConfirmationUIResult
from haystack_experimental.components.agents.human_in_the_loop.user_interfaces import RichConsoleUI, SimpleConsoleUI


def multiply_tool(x: int) -> int:
    return x * 2


@pytest.fixture
def tool():
    return create_tool_from_function(
        function=multiply_tool,
        name="test_tool",
        description="A test tool that multiplies input by 2.",
    )


class TestRichConsoleUI:
    @pytest.mark.parametrize(
        "choice,expected",
        [
            ("y", "confirm"),
            ("n", "reject"),
        ],
    )
    def test_process_choice(self, tool, choice, expected):
        ui = RichConsoleUI(console=MagicMock())
        params = {"x": 1}

        with patch(
            "haystack_experimental.components.agents.human_in_the_loop.user_interfaces.Prompt.ask",
            side_effect=[choice, "feedback"],
        ):
            result = ui.get_user_confirmation(tool.name, tool.description, params)

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == expected
        if expected == "reject":
            assert result.feedback == "feedback"
        if expected == "modify":
            assert result.new_tool_params == {"x": 2}

    def test_process_choice_modify(self, tool):
        ui = RichConsoleUI(console=MagicMock())
        params = {"x": 1}

        with patch(
            "haystack_experimental.components.agents.human_in_the_loop.user_interfaces.Prompt.ask",
            side_effect=["m", "2"],
        ):
            result = ui.get_user_confirmation(tool.name, tool.description, params)

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == "modify"
        assert result.new_tool_params == {"x": 2}

    def test_to_dict(self):
        ui = RichConsoleUI()
        data = ui.to_dict()
        assert data["type"] == (
            "haystack_experimental.components.agents.human_in_the_loop.user_interfaces.RichConsoleUI"
        )
        assert data["init_parameters"]["console"] is None

    def test_from_dict(self):
        ui = RichConsoleUI()
        data = ui.to_dict()
        new_ui = RichConsoleUI.from_dict(data)
        assert isinstance(new_ui, RichConsoleUI)


class TestSimpleConsoleUI:
    @pytest.mark.parametrize(
        "choice,expected",
        [
            ("y", "confirm"),
            ("yes", "confirm"),
            ("n", "reject"),
            ("m", "modify"),
        ],
    )
    def test_process_choice(self, tool, choice, expected):
        ui = SimpleConsoleUI()
        params = {"y": "abc"}

        inputs = {
            "n": ["feedback"],
            "m": ["new_value"],
        }.get(choice, [])

        with patch("builtins.input", side_effect=[choice] + inputs):
            result = ui.get_user_confirmation(tool.name, tool.description, params)

        assert isinstance(result, ConfirmationUIResult)
        assert result.action == expected
        if expected == "reject":
            assert result.feedback == "feedback"
        if expected == "modify":
            assert result.new_tool_params == {"y": "new_value"}

    def test_to_dict(self):
        ui = SimpleConsoleUI()
        data = ui.to_dict()
        assert data["type"] == (
            "haystack_experimental.components.agents.human_in_the_loop.user_interfaces.SimpleConsoleUI"
        )
        assert data["init_parameters"] == {}

    def test_from_dict(self):
        ui = SimpleConsoleUI()
        data = ui.to_dict()
        new_ui = SimpleConsoleUI.from_dict(data)
        assert isinstance(new_ui, SimpleConsoleUI)
