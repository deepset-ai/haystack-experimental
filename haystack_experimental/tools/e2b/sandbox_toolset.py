# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any

from haystack import logging
from haystack.lazy_imports import LazyImport
from haystack.tools import Tool, Toolset
from haystack.utils import Secret, deserialize_secrets_inplace

with LazyImport(message="Run 'pip install e2b'") as e2b_import:
    from e2b import Sandbox

logger = logging.getLogger(__name__)


@dataclass
class E2BSandboxToolset(Toolset):
    """
    A Haystack Toolset that provides bash command execution and filesystem access inside an E2B sandbox.

    E2BSandboxToolset creates and manages a connection to an E2B sandbox, exposing
    the following tools to a Haystack Agent:

    - **run_bash_command**: Execute arbitrary bash commands and capture stdout/stderr.
    - **read_file**: Read the content of a file from the sandbox filesystem.
    - **write_file**: Write content to a file in the sandbox filesystem.
    - **list_directory**: List the contents of a directory in the sandbox.

    The sandbox connection is established lazily via `warm_up()`, which is called
    automatically when the toolset is used within a Haystack pipeline or agent. The
    sandbox stays open for the configured `timeout` period of inactivity.

    ### Usage example

    ```python
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    from haystack_experimental.components.agents import Agent
    from haystack_experimental.tools.e2b import E2BSandboxToolset

    # Create the toolset – the sandbox connection is established during warm_up
    sandbox_toolset = E2BSandboxToolset(
        api_key=Secret.from_env_var("E2B_API_KEY"),
        sandbox_template="base",
        timeout=300,
    )

    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-4o"),
        tools=[sandbox_toolset],
    )
    ```

    The `warm_up()` call is handled automatically when the agent's pipeline starts, so
    you generally do not need to call it manually. If you use the toolset standalone,
    call `warm_up()` before the first tool invocation:

    ```python
    sandbox_toolset.warm_up()
    result = sandbox_toolset["run_bash_command"].invoke(command="echo hello")
    sandbox_toolset.close()
    ```
    """

    api_key: Secret = field(default_factory=lambda: Secret.from_env_var("E2B_API_KEY"))
    sandbox_template: str = field(default="base")
    timeout: int = field(default=300)
    environment_vars: dict[str, str] = field(default_factory=dict)

    # Private – not part of the public interface / serialized state
    _sandbox: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """
        Build the Tool objects that wrap the sandbox operations.

        The actual sandbox connection is deferred to `warm_up()`.
        """
        # Build tool list referencing bound methods on this instance so that
        # every tool shares the same sandbox connection.
        tools = [
            Tool(
                name="run_bash_command",
                description=(
                    "Execute a bash command inside the E2B sandbox and return the combined stdout, "
                    "stderr, and exit code. Use this to run shell scripts, install packages, compile "
                    "code, or perform any system-level operation."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": (
                                "Maximum number of seconds to wait for the command to finish. "
                                "Defaults to 60 seconds."
                            ),
                        },
                    },
                    "required": ["command"],
                },
                function=self._run_bash_command,
            ),
            Tool(
                name="read_file",
                description=(
                    "Read the text content of a file from the E2B sandbox filesystem and return it "
                    "as a string. The file must exist; use list_directory to verify paths first."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path of the file to read.",
                        },
                    },
                    "required": ["path"],
                },
                function=self._read_file,
            ),
            Tool(
                name="write_file",
                description=(
                    "Write text content to a file in the E2B sandbox filesystem. "
                    "Parent directories are created automatically if they do not exist. "
                    "Existing files are overwritten."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path of the file to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Text content to write into the file.",
                        },
                    },
                    "required": ["path", "content"],
                },
                function=self._write_file,
            ),
            Tool(
                name="list_directory",
                description=(
                    "List the files and subdirectories inside a directory in the E2B sandbox "
                    "filesystem. Returns a newline-separated list of names with a trailing '/' "
                    "appended to subdirectory names."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute or relative path of the directory to list.",
                        },
                    },
                    "required": ["path"],
                },
                function=self._list_directory,
            ),
        ]
        # Initialise the parent Toolset with our tools.
        # We bypass super().__post_init__ duplicate-name check by assigning directly so
        # that we can call the parent __post_init__ with the correct list in place.
        self.tools = tools
        super().__post_init__()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def warm_up(self) -> None:
        """
        Establish the connection to the E2B sandbox.

        This method is called automatically by the Haystack pipeline before the first
        tool invocation. It is idempotent – calling it multiple times has no effect if
        the sandbox is already running.

        :raises RuntimeError: If the E2B sandbox cannot be created (e.g. invalid API key
            or network error).
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

        Call this method when you are done using the toolset to avoid leaving
        idle sandboxes running and incurring unnecessary costs.
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
        Serialize the toolset configuration to a dictionary.

        The sandbox instance itself is not serialised; a fresh connection is
        established when `warm_up()` is called after deserialisation.

        :returns: Dictionary containing the serialised toolset configuration.
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
    def from_dict(cls, data: dict[str, Any]) -> "E2BSandboxToolset":
        """
        Deserialize an E2BSandboxToolset from a dictionary.

        :param data: Dictionary created by `to_dict()`.
        :returns: A new E2BSandboxToolset instance ready to be warmed up.
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
    # Private tool implementations
    # ------------------------------------------------------------------

    def _require_sandbox(self) -> "Sandbox":
        """Return the active sandbox or raise a helpful error if warm_up was not called."""
        if self._sandbox is None:
            raise RuntimeError(
                "E2B sandbox is not running. Call warm_up() before using the toolset, "
                "or add the toolset to a Haystack pipeline/agent which calls warm_up() automatically."
            )
        return self._sandbox

    def _run_bash_command(self, command: str, timeout: int = 60) -> str:
        """
        Execute a bash command in the sandbox.

        :param command: The bash command to run.
        :param timeout: Seconds to wait before killing the process.
        :returns: A formatted string containing exit_code, stdout and stderr.
        """
        sandbox = self._require_sandbox()
        try:
            result = sandbox.commands.run(command, timeout=timeout)
            return (
                f"exit_code: {result.exit_code}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to run bash command: {e}") from e

    def _read_file(self, path: str) -> str:
        """
        Read a file from the sandbox filesystem.

        :param path: Path to the file.
        :returns: The text content of the file.
        """
        sandbox = self._require_sandbox()
        try:
            content = sandbox.files.read(path)
            # e2b may return bytes; decode if necessary
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="replace")
            return str(content)
        except Exception as e:
            raise RuntimeError(f"Failed to read file '{path}': {e}") from e

    def _write_file(self, path: str, content: str) -> str:
        """
        Write content to a file in the sandbox filesystem.

        :param path: Destination path inside the sandbox.
        :param content: Text to write.
        :returns: A confirmation message with the file path.
        """
        sandbox = self._require_sandbox()
        try:
            sandbox.files.write(path, content)
            return f"File written successfully: {path}"
        except Exception as e:
            raise RuntimeError(f"Failed to write file '{path}': {e}") from e

    def _list_directory(self, path: str) -> str:
        """
        List the contents of a directory in the sandbox filesystem.

        :param path: Directory path to list.
        :returns: Newline-separated list of entries (directories end with '/').
        """
        sandbox = self._require_sandbox()
        try:
            entries = sandbox.files.list(path)
            lines = []
            for entry in entries:
                name = entry.name
                # Mark directories with a trailing slash for clarity
                if getattr(entry, "is_dir", False) or getattr(entry, "type", "") == "dir":
                    name = name + "/"
                lines.append(name)
            return "\n".join(lines) if lines else "(empty directory)"
        except Exception as e:
            raise RuntimeError(f"Failed to list directory '{path}': {e}") from e
