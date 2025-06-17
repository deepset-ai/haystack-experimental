# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack_experimental.core.pipeline import Pipeline


def test_agent_in_pipeline():
    pass
    # ToDo: pattern for breakpoints inside pipeline
    pipeline = Pipeline()
    data = {}
    pipeline.run(
         data, pipeline_breakpoint={("agent", 0, "chatgenerator", 0)}, debug_path="saved_states"
     )
    pipeline.run(
         data, pipeline_breakpoint={("agent", 0, "tool_invoker", 0, "web_search_tool", 0)}, debug_path="saved_states"
    )
