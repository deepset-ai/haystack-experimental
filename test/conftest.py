# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Generator

import pytest

from test.tracing.utils import SpyingTracer

from haystack.testing.test_utils import set_all_seeds
from haystack import tracing

set_all_seeds(0)


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"

@pytest.fixture()
def spying_tracer() -> Generator[SpyingTracer, None, None]:
    tracer = SpyingTracer()
    tracing.enable_tracing(tracer)

    yield tracer

    # Make sure to disable tracing after the test to avoid affecting other tests
    tracing.disable_tracing()
