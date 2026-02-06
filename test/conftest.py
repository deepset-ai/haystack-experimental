# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from unittest.mock import Mock, patch
from haystack.testing.test_utils import set_all_seeds

set_all_seeds(0)


@pytest.fixture()
def test_files_path():
    return Path(__file__).parent / "test_files"

@pytest.fixture
def mock_mem0_memory_client():
    """Mock the Mem0 MemoryClient."""
    with patch("haystack_experimental.memory_stores.mem0.memory_store.MemoryClient") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client
