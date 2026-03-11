# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack.utils import Secret

from haystack_experimental.tools.e2b.sandbox_toolset import E2BSandboxToolset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toolset(**kwargs) -> E2BSandboxToolset:
    """Create an E2BSandboxToolset with a dummy API key for testing."""
    defaults = {
        "api_key": Secret.from_token("test-api-key"),
        "sandbox_template": "base",
        "timeout": 120,
        "environment_vars": {},
    }
    defaults.update(kwargs)
    return E2BSandboxToolset(**defaults)


def _make_sandbox_mock() -> MagicMock:
    """Return a MagicMock that mimics the e2b Sandbox object."""
    sandbox = MagicMock()
    sandbox.sandbox_id = "sandbox-test-123"
    return sandbox


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestE2BSandboxToolsetInit:
    def test_default_parameters(self):
        toolset = _make_toolset()
        assert toolset.sandbox_template == "base"
        assert toolset.timeout == 120
        assert toolset.environment_vars == {}
        assert toolset._sandbox is None

    def test_custom_parameters(self):
        toolset = _make_toolset(
            sandbox_template="my-template",
            timeout=600,
            environment_vars={"FOO": "bar"},
        )
        assert toolset.sandbox_template == "my-template"
        assert toolset.timeout == 600
        assert toolset.environment_vars == {"FOO": "bar"}

    def test_tools_are_created(self):
        toolset = _make_toolset()
        tool_names = {tool.name for tool in toolset}
        assert tool_names == {"run_bash_command", "read_file", "write_file", "list_directory"}

    def test_tools_have_descriptions(self):
        toolset = _make_toolset()
        for tool in toolset:
            assert tool.description, f"Tool '{tool.name}' has no description"

    def test_tools_have_valid_parameters_schema(self):
        toolset = _make_toolset()
        for tool in toolset:
            assert "type" in tool.parameters
            assert "properties" in tool.parameters

    def test_run_bash_command_required_parameter(self):
        toolset = _make_toolset()
        bash_tool = next(t for t in toolset if t.name == "run_bash_command")
        assert "command" in bash_tool.parameters["required"]

    def test_read_file_required_parameter(self):
        toolset = _make_toolset()
        read_tool = next(t for t in toolset if t.name == "read_file")
        assert "path" in read_tool.parameters["required"]

    def test_write_file_required_parameters(self):
        toolset = _make_toolset()
        write_tool = next(t for t in toolset if t.name == "write_file")
        assert "path" in write_tool.parameters["required"]
        assert "content" in write_tool.parameters["required"]

    def test_list_directory_required_parameter(self):
        toolset = _make_toolset()
        list_tool = next(t for t in toolset if t.name == "list_directory")
        assert "path" in list_tool.parameters["required"]

    def test_toolset_len(self):
        toolset = _make_toolset()
        assert len(toolset) == 4

    def test_toolset_contains_by_name(self):
        toolset = _make_toolset()
        assert "run_bash_command" in toolset
        assert "read_file" in toolset
        assert "write_file" in toolset
        assert "list_directory" in toolset
        assert "nonexistent_tool" not in toolset


# ---------------------------------------------------------------------------
# warm_up
# ---------------------------------------------------------------------------


class TestE2BSandboxToolsetWarmUp:
    @patch("haystack_experimental.tools.e2b.sandbox_toolset.e2b_import")
    @patch("haystack_experimental.tools.e2b.sandbox_toolset.Sandbox")
    def test_warm_up_creates_sandbox(self, mock_sandbox_cls, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_instance = _make_sandbox_mock()
        mock_sandbox_cls.return_value = mock_sandbox_instance

        toolset = _make_toolset()
        toolset.warm_up()

        mock_sandbox_cls.assert_called_once_with(
            api_key="test-api-key",
            template="base",
            timeout=120,
            envs=None,
        )
        assert toolset._sandbox is mock_sandbox_instance

    @patch("haystack_experimental.tools.e2b.sandbox_toolset.e2b_import")
    @patch("haystack_experimental.tools.e2b.sandbox_toolset.Sandbox")
    def test_warm_up_passes_environment_vars(self, mock_sandbox_cls, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_cls.return_value = _make_sandbox_mock()

        toolset = _make_toolset(environment_vars={"MY_VAR": "value"})
        toolset.warm_up()

        _, kwargs = mock_sandbox_cls.call_args
        assert kwargs["envs"] == {"MY_VAR": "value"}

    @patch("haystack_experimental.tools.e2b.sandbox_toolset.e2b_import")
    @patch("haystack_experimental.tools.e2b.sandbox_toolset.Sandbox")
    def test_warm_up_is_idempotent(self, mock_sandbox_cls, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_cls.return_value = _make_sandbox_mock()

        toolset = _make_toolset()
        toolset.warm_up()
        toolset.warm_up()

        # Sandbox should only be created once
        mock_sandbox_cls.assert_called_once()

    @patch("haystack_experimental.tools.e2b.sandbox_toolset.e2b_import")
    @patch("haystack_experimental.tools.e2b.sandbox_toolset.Sandbox")
    def test_warm_up_raises_on_sandbox_error(self, mock_sandbox_cls, mock_e2b_import):
        mock_e2b_import.check.return_value = None
        mock_sandbox_cls.side_effect = Exception("connection refused")

        toolset = _make_toolset()
        with pytest.raises(RuntimeError, match="Failed to start E2B sandbox"):
            toolset.warm_up()


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestE2BSandboxToolsetClose:
    def test_close_without_warm_up_is_noop(self):
        toolset = _make_toolset()
        toolset.close()  # must not raise
        assert toolset._sandbox is None

    def test_close_kills_sandbox(self):
        toolset = _make_toolset()
        mock_sandbox = _make_sandbox_mock()
        toolset._sandbox = mock_sandbox

        toolset.close()

        mock_sandbox.kill.assert_called_once()
        assert toolset._sandbox is None

    def test_close_clears_sandbox_on_kill_error(self):
        toolset = _make_toolset()
        mock_sandbox = _make_sandbox_mock()
        mock_sandbox.kill.side_effect = Exception("kill failed")
        toolset._sandbox = mock_sandbox

        toolset.close()  # must not raise

        assert toolset._sandbox is None


# ---------------------------------------------------------------------------
# Tool invocations
# ---------------------------------------------------------------------------


class TestE2BSandboxToolsetRunBashCommand:
    def _toolset_with_sandbox(self) -> tuple[E2BSandboxToolset, MagicMock]:
        toolset = _make_toolset()
        mock_sandbox = _make_sandbox_mock()
        toolset._sandbox = mock_sandbox
        return toolset, mock_sandbox

    def test_run_bash_command_returns_formatted_output(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_result = MagicMock()
        mock_result.exit_code = 0
        mock_result.stdout = "hello world\n"
        mock_result.stderr = ""
        mock_sandbox.commands.run.return_value = mock_result

        output = toolset._run_bash_command("echo hello world")

        assert "exit_code: 0" in output
        assert "hello world" in output
        mock_sandbox.commands.run.assert_called_once_with("echo hello world", timeout=60)

    def test_run_bash_command_passes_custom_timeout(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.commands.run.return_value = MagicMock(exit_code=0, stdout="", stderr="")

        toolset._run_bash_command("sleep 5", timeout=30)

        mock_sandbox.commands.run.assert_called_once_with("sleep 5", timeout=30)

    def test_run_bash_command_raises_when_no_sandbox(self):
        toolset = _make_toolset()
        with pytest.raises(RuntimeError, match="E2B sandbox is not running"):
            toolset._run_bash_command("ls")

    def test_run_bash_command_wraps_sandbox_exception(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.commands.run.side_effect = Exception("timeout")

        with pytest.raises(RuntimeError, match="Failed to run bash command"):
            toolset._run_bash_command("sleep 1000")


class TestE2BSandboxToolsetReadFile:
    def _toolset_with_sandbox(self) -> tuple[E2BSandboxToolset, MagicMock]:
        toolset = _make_toolset()
        mock_sandbox = _make_sandbox_mock()
        toolset._sandbox = mock_sandbox
        return toolset, mock_sandbox

    def test_read_file_returns_string(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.files.read.return_value = "file content"

        result = toolset._read_file("/some/file.txt")

        assert result == "file content"
        mock_sandbox.files.read.assert_called_once_with("/some/file.txt")

    def test_read_file_decodes_bytes(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.files.read.return_value = b"binary content"

        result = toolset._read_file("/binary.bin")

        assert result == "binary content"

    def test_read_file_raises_when_no_sandbox(self):
        toolset = _make_toolset()
        with pytest.raises(RuntimeError, match="E2B sandbox is not running"):
            toolset._read_file("/some/file.txt")

    def test_read_file_wraps_sandbox_exception(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.files.read.side_effect = Exception("file not found")

        with pytest.raises(RuntimeError, match="Failed to read file"):
            toolset._read_file("/nonexistent.txt")


class TestE2BSandboxToolsetWriteFile:
    def _toolset_with_sandbox(self) -> tuple[E2BSandboxToolset, MagicMock]:
        toolset = _make_toolset()
        mock_sandbox = _make_sandbox_mock()
        toolset._sandbox = mock_sandbox
        return toolset, mock_sandbox

    def test_write_file_returns_confirmation(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()

        result = toolset._write_file("/output/result.txt", "hello")

        assert "/output/result.txt" in result
        mock_sandbox.files.write.assert_called_once_with("/output/result.txt", "hello")

    def test_write_file_raises_when_no_sandbox(self):
        toolset = _make_toolset()
        with pytest.raises(RuntimeError, match="E2B sandbox is not running"):
            toolset._write_file("/some/path.txt", "content")

    def test_write_file_wraps_sandbox_exception(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.files.write.side_effect = Exception("permission denied")

        with pytest.raises(RuntimeError, match="Failed to write file"):
            toolset._write_file("/protected/file.txt", "data")


class TestE2BSandboxToolsetListDirectory:
    def _toolset_with_sandbox(self) -> tuple[E2BSandboxToolset, MagicMock]:
        toolset = _make_toolset()
        mock_sandbox = _make_sandbox_mock()
        toolset._sandbox = mock_sandbox
        return toolset, mock_sandbox

    def _make_entry(self, name: str, is_dir: bool = False) -> MagicMock:
        entry = MagicMock()
        entry.name = name
        entry.is_dir = is_dir
        return entry

    def test_list_directory_returns_names(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.files.list.return_value = [
            self._make_entry("file.txt"),
            self._make_entry("subdir", is_dir=True),
        ]

        result = toolset._list_directory("/home/user")

        assert "file.txt" in result
        assert "subdir/" in result
        mock_sandbox.files.list.assert_called_once_with("/home/user")

    def test_list_directory_empty(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.files.list.return_value = []

        result = toolset._list_directory("/empty")

        assert result == "(empty directory)"

    def test_list_directory_raises_when_no_sandbox(self):
        toolset = _make_toolset()
        with pytest.raises(RuntimeError, match="E2B sandbox is not running"):
            toolset._list_directory("/home")

    def test_list_directory_wraps_sandbox_exception(self):
        toolset, mock_sandbox = self._toolset_with_sandbox()
        mock_sandbox.files.list.side_effect = Exception("not a directory")

        with pytest.raises(RuntimeError, match="Failed to list directory"):
            toolset._list_directory("/nonexistent")


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestE2BSandboxToolsetSerialisation:
    """Serialisation tests use env-var secrets (the only serialisable Secret type)."""

    def _make_env_toolset(self, **kwargs) -> E2BSandboxToolset:
        defaults = {
            "api_key": Secret.from_env_var("E2B_API_KEY"),
            "sandbox_template": "base",
            "timeout": 120,
            "environment_vars": {},
        }
        defaults.update(kwargs)
        return E2BSandboxToolset(**defaults)

    def test_to_dict_contains_expected_keys(self):
        toolset = self._make_env_toolset(sandbox_template="my-template", timeout=600)
        data = toolset.to_dict()

        assert "type" in data
        assert "data" in data
        assert data["data"]["sandbox_template"] == "my-template"
        assert data["data"]["timeout"] == 600

    def test_to_dict_does_not_include_sandbox_instance(self):
        toolset = self._make_env_toolset()
        toolset._sandbox = _make_sandbox_mock()  # simulate warm-up
        data = toolset.to_dict()

        assert "_sandbox" not in data["data"]
        assert "sandbox" not in data["data"]

    def test_from_dict_round_trip(self):
        original = self._make_env_toolset(
            sandbox_template="custom",
            timeout=900,
            environment_vars={"KEY": "value"},
        )
        data = original.to_dict()
        restored = E2BSandboxToolset.from_dict(data)

        assert restored.sandbox_template == "custom"
        assert restored.timeout == 900
        assert restored.environment_vars == {"KEY": "value"}
        assert restored._sandbox is None  # sandbox not restored

    def test_from_dict_creates_tools(self):
        original = self._make_env_toolset()
        data = original.to_dict()
        restored = E2BSandboxToolset.from_dict(data)

        assert len(restored) == 4
        assert "run_bash_command" in restored

    def test_to_dict_type_is_qualified_class_name(self):
        toolset = self._make_env_toolset()
        data = toolset.to_dict()
        assert "E2BSandboxToolset" in data["type"]
