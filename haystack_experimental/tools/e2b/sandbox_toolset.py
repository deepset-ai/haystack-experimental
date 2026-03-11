# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.tools import Tool
from haystack.utils import Secret

from haystack_experimental.tools.e2b.bash_tool import RunBashCommandTool
from haystack_experimental.tools.e2b.e2b_sandbox import E2BSandbox
from haystack_experimental.tools.e2b.list_directory_tool import ListDirectoryTool
from haystack_experimental.tools.e2b.read_file_tool import ReadFileTool
from haystack_experimental.tools.e2b.write_file_tool import WriteFileTool


def create_e2b_tools(
    api_key: Secret | None = None,
    sandbox_template: str = "base",
    timeout: int = 300,
    environment_vars: dict[str, str] | None = None,
) -> tuple[E2BSandbox, list[Tool]]:
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
