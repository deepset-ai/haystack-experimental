# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class AsyncPipeline:
    """
    Asynchronous version of the orchestration engine.

    Note: Currently not supported in haystack-experimental 0.5.0.
    Please use haystack-experimental 0.4.0 for async pipeline functionality.
    """

    def __init__(self, **kwargs):
        raise RuntimeError(
            "The haystack-experimental 0.5.0 release does not support async pipelines. "
            "Please use the haystack-experimental 0.4.0 release instead."
        )
