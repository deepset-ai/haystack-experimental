# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from haystack import logging
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install e2b'") as e2b_import:
    from e2b import Sandbox

logger = logging.getLogger(__name__)


@dataclass
class E2BSandbox:
    """
    Manages the lifecycle of an E2B cloud sandbox.

    Instantiate this class and pass it to one or more E2B tool classes
    (``RunBashCommandTool``, ``ReadFileTool``, ``WriteFileTool``,
    ``ListDirectoryTool``) to share a single sandbox environment across all
    tools.  All tools that receive the same ``E2BSandbox`` instance operate
    inside the same live sandbox process.

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    from haystack_experimental.components.agents import Agent
    from haystack_experimental.tools.e2b import (
        E2BSandbox,
        RunBashCommandTool,
        ReadFileTool,
        WriteFileTool,
        ListDirectoryTool,
    )

    sandbox = E2BSandbox()
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=[
            RunBashCommandTool(sandbox=sandbox),
            ReadFileTool(sandbox=sandbox),
            WriteFileTool(sandbox=sandbox),
            ListDirectoryTool(sandbox=sandbox),
        ],
    )
    ```

    Lifecycle is handled automatically by the Agent's pipeline. If you use the
    tools standalone, call :meth:`warm_up` before the first tool invocation:

    ```python
    sandbox.warm_up()
    # … use tools …
    sandbox.close()
    ```
    """

    api_key: Secret = field(default_factory=lambda: Secret.from_env_var("E2B_API_KEY"))
    sandbox_template: str = field(default="base")
    timeout: int = field(default=300)
    environment_vars: dict[str, str] = field(default_factory=dict)

    # Private – not serialised
    _sandbox: Any = field(default=None, init=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Establish the connection to the E2B sandbox.

        Idempotent – calling it multiple times has no effect if the sandbox is
        already running.

        :raises RuntimeError: If the E2B sandbox cannot be created.
        """
        if self._sandbox is not None:
            return

        e2b_import.check()
        resolved_key = self.api_key.resolve_value()
        try:
            logger.info(
                "Starting E2B sandbox (template={template}, timeout={timeout}s)",
                template=self.sandbox_template,
                timeout=self.timeout,
            )
            self._sandbox = Sandbox(
                api_key=resolved_key,
                template=self.sandbox_template,
                timeout=self.timeout,
                envs=self.environment_vars if self.environment_vars else None,
            )
            logger.info("E2B sandbox started (id={sandbox_id})", sandbox_id=self._sandbox.sandbox_id)
        except Exception as e:
            raise RuntimeError(f"Failed to start E2B sandbox: {e}") from e

    def close(self) -> None:
        """
        Shut down the E2B sandbox and release all associated resources.

        Call this when you are done to avoid leaving idle sandboxes running.
        """
        if self._sandbox is None:
            return
        try:
            self._sandbox.kill()
            logger.info("E2B sandbox closed")
        except Exception as e:
            logger.warning("Failed to close E2B sandbox: {error}", error=e)
        finally:
            self._sandbox = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the sandbox configuration to a dictionary.

        :returns: Dictionary containing the serialised configuration.
        """
        from haystack.core.serialization import generate_qualified_class_name

        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {
                "api_key": self.api_key.to_dict(),
                "sandbox_template": self.sandbox_template,
                "timeout": self.timeout,
                "environment_vars": self.environment_vars,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "E2BSandbox":
        """
        Deserialize an :class:`E2BSandbox` from a dictionary.

        :param data: Dictionary created by :meth:`to_dict`.
        :returns: A new :class:`E2BSandbox` instance ready to be warmed up.
        """
        inner = data["data"]
        deserialize_secrets_inplace(inner, keys=["api_key"])
        return cls(
            api_key=inner["api_key"],
            sandbox_template=inner.get("sandbox_template", "base"),
            timeout=inner.get("timeout", 300),
            environment_vars=inner.get("environment_vars", {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers (used by the tool classes)
    # ------------------------------------------------------------------

    def _require_sandbox(self) -> "Sandbox":
        """Return the active sandbox or raise a helpful error."""
        if self._sandbox is None:
            raise RuntimeError(
                "E2B sandbox is not running. Call warm_up() before using the tools, "
                "or add the sandbox to a Haystack pipeline/agent which calls warm_up() automatically."
            )
        return self._sandbox


# ---------------------------------------------------------------------------
# Tool classes
# ---------------------------------------------------------------------------


class RunBashCommandTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that executes bash commands inside an E2B sandbox.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    sandbox = E2BSandbox()
    bash_tool = RunBashCommandTool(sandbox=sandbox)
    read_tool = ReadFileTool(sandbox=sandbox)
    agent = Agent(chat_generator=..., tools=[bash_tool, read_tool])
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        :param sandbox: The :class:`E2BSandbox` instance that will execute commands.
        """

        def run_bash_command(command: str, timeout: int = 60) -> str:
            sb = sandbox._require_sandbox()
            try:
                result = sb.commands.run(command, timeout=timeout)
                return f"exit_code: {result.exit_code}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            except Exception as e:
                raise RuntimeError(f"Failed to run bash command: {e}") from e

        super().__init__(
            name="run_bash_command",
            description=(
                "Execute a bash command inside the E2B sandbox and return the combined stdout, "
                "stderr, and exit code. Use this to run shell scripts, install packages, compile "
                "code, or perform any system-level operation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."},
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Maximum number of seconds to wait for the command to finish. Defaults to 60 seconds."
                        ),
                    },
                },
                "required": ["command"],
            },
            function=run_bash_command,
        )
        self._e2b_sandbox = sandbox


class ReadFileTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that reads files from an E2B sandbox filesystem.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    sandbox = E2BSandbox()
    read_tool = ReadFileTool(sandbox=sandbox)
    agent = Agent(chat_generator=..., tools=[read_tool])
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        :param sandbox: The :class:`E2BSandbox` instance to read files from.
        """

        def read_file(path: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                content = sb.files.read(path)
                if isinstance(content, bytes):
                    return content.decode("utf-8", errors="replace")
                return str(content)
            except Exception as e:
                raise RuntimeError(f"Failed to read file '{path}': {e}") from e

        super().__init__(
            name="read_file",
            description=(
                "Read the text content of a file from the E2B sandbox filesystem and return it "
                "as a string. The file must exist; use list_directory to verify paths first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path of the file to read."}
                },
                "required": ["path"],
            },
            function=read_file,
        )
        self._e2b_sandbox = sandbox


class WriteFileTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that writes files to an E2B sandbox filesystem.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    sandbox = E2BSandbox()
    write_tool = WriteFileTool(sandbox=sandbox)
    agent = Agent(chat_generator=..., tools=[write_tool])
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        :param sandbox: The :class:`E2BSandbox` instance to write files to.
        """

        def write_file(path: str, content: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                sb.files.write(path, content)
                return f"File written successfully: {path}"
            except Exception as e:
                raise RuntimeError(f"Failed to write file '{path}': {e}") from e

        super().__init__(
            name="write_file",
            description=(
                "Write text content to a file in the E2B sandbox filesystem. "
                "Parent directories are created automatically if they do not exist. "
                "Existing files are overwritten."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path of the file to write."},
                    "content": {"type": "string", "description": "Text content to write into the file."},
                },
                "required": ["path", "content"],
            },
            function=write_file,
        )
        self._e2b_sandbox = sandbox


class ListDirectoryTool(Tool):
    """
    A :class:`~haystack.tools.Tool` that lists directory contents in an E2B sandbox.

    Pass the same :class:`E2BSandbox` instance to multiple tool classes so they
    all operate in the same live sandbox environment.

    ### Usage example

    ```python
    sandbox = E2BSandbox()
    list_tool = ListDirectoryTool(sandbox=sandbox)
    agent = Agent(chat_generator=..., tools=[list_tool])
    ```
    """

    def __init__(self, sandbox: E2BSandbox) -> None:
        """
        :param sandbox: The :class:`E2BSandbox` instance to list directories from.
        """

        def list_directory(path: str) -> str:
            sb = sandbox._require_sandbox()
            try:
                entries = sb.files.list(path)
                lines = []
                for entry in entries:
                    name = entry.name
                    if getattr(entry, "is_dir", False) or getattr(entry, "type", "") == "dir":
                        name = name + "/"
                    lines.append(name)
                return "\n".join(lines) if lines else "(empty directory)"
            except Exception as e:
                raise RuntimeError(f"Failed to list directory '{path}': {e}") from e

        super().__init__(
            name="list_directory",
            description=(
                "List the files and subdirectories inside a directory in the E2B sandbox "
                "filesystem. Returns a newline-separated list of names with a trailing '/' "
                "appended to subdirectory names."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path of the directory to list."}
                },
                "required": ["path"],
            },
            function=list_directory,
        )
        self._e2b_sandbox = sandbox


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_e2b_tools(
    api_key: Secret | None = None,
    sandbox_template: str = "base",
    timeout: int = 300,
    environment_vars: dict[str, str] | None = None,
) -> tuple["E2BSandbox", list[Tool]]:
    """
    Create an :class:`E2BSandbox` and all four E2B tools in one call.

    Returns both the sandbox (for lifecycle management) and the list of tools
    so that callers can pass any subset of the tools to an Agent.

    :param api_key: E2B API key. Defaults to ``Secret.from_env_var("E2B_API_KEY")``.
    :param sandbox_template: E2B sandbox template name. Defaults to ``"base"``.
    :param timeout: Sandbox inactivity timeout in seconds. Defaults to ``300``.
    :param environment_vars: Optional environment variables to inject into the sandbox.
    :returns: A ``(sandbox, tools)`` tuple where *tools* is a list of four
        :class:`~haystack.tools.Tool` objects: ``run_bash_command``, ``read_file``,
        ``write_file``, and ``list_directory``.

    ### Usage example

    ```python
    from haystack.utils import Secret
    from haystack_experimental.tools.e2b import create_e2b_tools

    sandbox, tools = create_e2b_tools(
        api_key=Secret.from_env_var("E2B_API_KEY"),
    )

    # Use all four tools:
    agent = Agent(chat_generator=..., tools=tools)

    # Or only a subset – they still share the same sandbox connection,
    # so run_bash_command and read_file operate inside the same environment:
    bash_tool, read_tool = tools[0], tools[1]
    agent = Agent(chat_generator=..., tools=[bash_tool, read_tool])
    ```
    """
    if api_key is None:
        api_key = Secret.from_env_var("E2B_API_KEY")
    sandbox = E2BSandbox(
        api_key=api_key, sandbox_template=sandbox_template, timeout=timeout, environment_vars=environment_vars or {}
    )
    tools: list[Tool] = [
        RunBashCommandTool(sandbox=sandbox),
        ReadFileTool(sandbox=sandbox),
        WriteFileTool(sandbox=sandbox),
        ListDirectoryTool(sandbox=sandbox),
    ]
    return sandbox, tools
