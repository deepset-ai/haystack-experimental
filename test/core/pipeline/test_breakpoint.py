# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import pytest

from haystack.components.joiners import BranchJoiner
from haystack_experimental.core.pipeline.pipeline import Pipeline
from haystack_experimental.core.pipeline.breakpoint import _transform_json_structure, _serialize_component_input, _deserialize_component_input, load_state, _validate_breakpoint, _validate_resume_state


class TestBreakpoint:
    """
    This class contains only unit tests for the breakpoint module.
    """

def test_transform_json_structure_unwraps_sender_value():
    data = {
        "key1": [{"sender": None, "value": "some value"}],
        "key2": [{"sender": "comp1", "value": 42}],
        "key3": "direct value"
    }

    result = _transform_json_structure(data)

    assert result == {
        "key1": "some value",
        "key2": 42,
        "key3": "direct value"
    }

def test_transform_json_structure_handles_nested_structures():
    data = {
        "key1": [{"sender": None, "value": "value1"}],
        "key2": {
            "nested": [{"sender": "comp1", "value": "value2"}],
            "direct": "value3"
        },
        "key3": [
            [{"sender": None, "value": "value4"}],
            [{"sender": "comp2", "value": "value5"}]
        ]
    }

    result = _transform_json_structure(data)

    assert result == {
        "key1": "value1",
        "key2": {
            "nested": "value2",
            "direct": "value3"
        },
        "key3": [
            "value4",
            "value5"
        ]
    }

def test_serialize_component_input_handles_objects_with_to_dict():
    class TestObject:
        def __init__(self, value):
            self.value = value

        def to_dict(self):
            return {"value": self.value}

    obj = TestObject("test")
    result = _serialize_component_input(obj)
    assert result == {
        "_type": "TestObject",
        "value": "test"
    }

def test_serialize_component_input_handles_objects_without_to_dict():
    class TestObject:
        def __init__(self, value):
            self.value = value

    obj = TestObject("test")
    result = _serialize_component_input(obj)
    assert result == {
        "_type": "TestObject",
        "attributes": {"value": "test"}
    }

def test_serialize_component_input_handles_nested_structures():
    class TestObject:
        def __init__(self, value):
            self.value = value

        def to_dict(self):
            return {"value": self.value}

    obj = TestObject("test")
    data = {
        "key1": obj,
        "key2": [obj, "string"],
        "key3": {"nested": obj}
    }
    result = _serialize_component_input(data)

    assert result["key1"]["_type"] == "TestObject"
    assert result["key2"][0]["_type"] == "TestObject"
    assert result["key2"][1] == "string"
    assert result["key3"]["nested"]["_type"] == "TestObject"

def test_deserialize_component_input_handles_primitive_types():
    data = {
        "string": "test",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None
    }
    result = _deserialize_component_input(data)
    assert result == data

def test_deserialize_component_input_handles_lists():
    data = {
        "primitive_list": [1, 2, 3],
        "mixed_list": [1, "string", True]
    }
    result = _deserialize_component_input(data)
    assert result == data

def test_deserialize_component_input_handles_dicts():
    data = {
        "key1": "value1",
        "key2": {"nested": "value2"}
    }
    result = _deserialize_component_input(data)
    assert result == data

def test_deserialize_component_input_handles_nested_lists():
    """Test that _deserialize_component_input handles nested lists"""
    data = {
        "nested_list": [[1, 2], [3, 4]],
        "mixed_nested": [[1, "string"], [True, 3.14]]
    }

    result = _deserialize_component_input(data)

    assert result == data

def test_deserialize_component_input_handles_nested_dicts():
    """Test that _deserialize_component_input handles nested dictionaries"""
    data = {
        "key1": {
            "nested1": "value1",
            "nested2": {
                "deep": "value2"
            }
        }
    }

    result = _deserialize_component_input(data)

    assert result == data

def test_deserialize_component_input_handles_empty_structures():
    """Test that _deserialize_component_input handles empty structures"""
    data = {
        "empty_list": [],
        "empty_dict": {},
        "nested_empty": {"empty": []}
    }

    result = _deserialize_component_input(data)

    assert result == data

def test_validate_resume_state_validates_required_keys():
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0}
        # Missing pipeline_state
    }

    with pytest.raises(ValueError, match="Invalid state file: missing required keys"):
        _validate_resume_state(state)

    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {}
            # Missing ordered_component_names
        }
    }

    with pytest.raises(ValueError, match="Invalid pipeline_state: missing required keys"):
        _validate_resume_state(state)

def test_validate_resume_state_validates_component_consistency():
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp3"]  # inconsistent with component_visits
        }
    }

    with pytest.raises(ValueError, match="Inconsistent state: components in pipeline_state"):
        _validate_resume_state(state)

def test_validate_resume_state_validates_valid_state():
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"]
        }
    }

    _validate_resume_state(state) # should not raise any exception

def test_load_state_loads_valid_state(tmp_path):
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"]
        }
    }
    state_file = tmp_path / "state.json"
    with open(state_file, "w") as f:
        json.dump(state, f)

    loaded_state = load_state(state_file)
    assert loaded_state == state

def test_load_state_handles_invalid_state(tmp_path):
    state = {
        "input_data": {},
        "pipeline_breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp3"]  # inconsistent with component_visits
        }
    }

    state_file = tmp_path / "invalid_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f)

    with pytest.raises(ValueError, match="Invalid pipeline state from"):
        load_state(state_file)
