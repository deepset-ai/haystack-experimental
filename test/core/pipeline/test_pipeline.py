# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
import json
import pytest

from haystack.components.joiners import BranchJoiner
from haystack.core.component import component
from haystack_experimental.core.errors import PipelineRuntimeError
from haystack_experimental.core.pipeline.pipeline import (
    Pipeline,
    transform_json_structure,
    serialize_component_input,
    deserialize_component_input
)


class TestPipeline:
    """
    This class contains only unit tests for the Pipeline class.
    It doesn't test Pipeline.run(), that is done separately in a different way.
    """

    def test_pipeline_thread_safety(self, waiting_component, spying_tracer):
        # Initialize pipeline with synchronous components
        pp = Pipeline()
        pp.add_component("wait", waiting_component())

        run_data = [{"wait_for": 1}, {"wait_for": 2}]

        # Use ThreadPoolExecutor to run pipeline calls in parallel
        with ThreadPoolExecutor(max_workers=len(run_data)) as executor:
            # Submit pipeline runs to the executor
            futures = [executor.submit(pp.run, data) for data in run_data]

            # Wait for all futures to complete
            for future in futures:
                future.result()

        # Verify component visits using tracer
        component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run"]

        for span in component_spans:
            assert span.tags["haystack.component.visits"] == 1

    def test_prepare_component_inputs(self):
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        component_name = "joiner_1"
        pp.add_component(component_name, joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect(component_name, "joiner_2")
        inputs = {"joiner_1": {"value": [{"sender": None, "value": "test_value"}]}}
        comp_dict = pp._get_component_with_graph_metadata_and_visits(component_name, 0)

        _ = pp._consume_component_inputs(component_name=component_name, component=comp_dict, inputs=inputs)
        # We remove input in greedy variadic sockets, even if they are from the user
        assert inputs == {"joiner_1": {}}

    def test__run_component_success(self):
        """Test successful component execution"""
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        component_name = "joiner_1"
        pp.add_component(component_name, joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect(component_name, "joiner_2")
        inputs = {"value": ["test_value"]}

        outputs = pp._run_component(
            component_name=component_name,
            component=pp._get_component_with_graph_metadata_and_visits(component_name, 0),
            inputs=inputs,
            component_visits={component_name: 0, "joiner_2": 0},
        )

        assert outputs == {"value": "test_value"}

    def test__run_component_fail(self):
        """Test error when component doesn't return a dictionary"""

        @component
        class WrongOutput:
            @component.output_types(output=str)
            def run(self, value: str):
                return "not_a_dict"

        wrong = WrongOutput()
        pp = Pipeline()
        pp.add_component("wrong", wrong)
        inputs = {"value": "test_value"}

        with pytest.raises(PipelineRuntimeError) as exc_info:
            pp._run_component(
                component_name="wrong",
                component=pp._get_component_with_graph_metadata_and_visits("wrong", 0),
                inputs=inputs,
                component_visits={"wrong": 0},
            )

        assert "Expected a dict" in str(exc_info.value)

    def test_run(self):
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        pp.add_component("joiner_1", joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect("joiner_1", "joiner_2")

        _ = pp.run({"value": "test_value"})

    def test_validate_breakpoints(self):
        # simple pipeline
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pipeline = Pipeline()
        pipeline.add_component("comp1", joiner_1)
        pipeline.add_component("comp2", joiner_2)
        pipeline.connect("comp1", "comp2")

        # valid breakpoints
        breakpoints = {("comp1", 0), ("comp2", 1)}
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == {("comp1", 0), ("comp2", 1)}

        # should default to 0
        breakpoints = {("comp1", None), ("comp2", 1)}
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == {("comp1", 0), ("comp2", 1)}

        # should remain as it is
        breakpoints = {("comp1", -1)}
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == {("comp1", -1)}

        # contains invalid components
        breakpoints = {("comp1", 0), ("non_existent_component", 1)}
        with pytest.raises(ValueError, match="Breakpoint .* is not a registered component"):
            pipeline._validate_breakpoints(breakpoints)

        # no breakpoints are defined
        breakpoints = set()
        validated = pipeline._validate_breakpoints(breakpoints)
        assert validated == set()


def test_transform_json_structure_unwraps_sender_value():
    data = {
        "key1": [{"sender": None, "value": "some value"}],
        "key2": [{"sender": "comp1", "value": 42}],
        "key3": "direct value"
    }

    result = transform_json_structure(data)

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

    result = transform_json_structure(data)

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
    result = serialize_component_input(obj)
    assert result == {
        "_type": "TestObject",
        "value": "test"
    }

def test_serialize_component_input_handles_objects_without_to_dict():
    class TestObject:
        def __init__(self, value):
            self.value = value

    obj = TestObject("test")
    result = serialize_component_input(obj)
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
    result = serialize_component_input(data)

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
    result = deserialize_component_input(data)
    assert result == data

def test_deserialize_component_input_handles_lists():
    data = {
        "primitive_list": [1, 2, 3],
        "mixed_list": [1, "string", True]
    }
    result = deserialize_component_input(data)
    assert result == data

def test_deserialize_component_input_handles_dicts():
    data = {
        "key1": "value1",
        "key2": {"nested": "value2"}
    }
    result = deserialize_component_input(data)
    assert result == data

def test_deserialize_component_input_handles_nested_lists():
    """Test that _deserialize_component_input handles nested lists"""
    data = {
        "nested_list": [[1, 2], [3, 4]],
        "mixed_nested": [[1, "string"], [True, 3.14]]
    }

    result = deserialize_component_input(data)

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

    result = deserialize_component_input(data)

    assert result == data

def test_deserialize_component_input_handles_empty_structures():
    """Test that _deserialize_component_input handles empty structures"""
    data = {
        "empty_list": [],
        "empty_dict": {},
        "nested_empty": {"empty": []}
    }

    result = deserialize_component_input(data)

    assert result == data

def test_validate_resume_state_validates_required_keys():
    state = {
        "input_data": {},
        "breakpoint": {"component": "comp1", "visits": 0}
        # Missing pipeline_state
    }

    with pytest.raises(ValueError, match="Invalid state file: missing required keys"):
        Pipeline._validate_resume_state(state)

    state = {
        "input_data": {},
        "breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {}
            # Missing ordered_component_names
        }
    }

    with pytest.raises(ValueError, match="Invalid pipeline_state: missing required keys"):
        Pipeline._validate_resume_state(state)

def test_validate_resume_state_validates_component_consistency():
    state = {
        "input_data": {},
        "breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp3"]  # inconsistent with component_visits
        }
    }

    with pytest.raises(ValueError, match="Inconsistent state: components in pipeline_state"):
        Pipeline._validate_resume_state(state)

def test_validate_resume_state_validates_valid_state():
    state = {
        "input_data": {},
        "breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"]
        }
    }

    Pipeline._validate_resume_state(state) # should not raise any exception

def test_load_state_loads_valid_state(tmp_path):
    state = {
        "input_data": {},
        "breakpoint": {"component": "comp1", "visits": 0},
        "pipeline_state": {
            "inputs": {},
            "component_visits": {"comp1": 0, "comp2": 0},
            "ordered_component_names": ["comp1", "comp2"]
        }
    }
    state_file = tmp_path / "state.json"
    with open(state_file, "w") as f:
        json.dump(state, f)

    loaded_state = Pipeline.load_state(state_file)
    assert loaded_state == state

def test_load_state_handles_invalid_state(tmp_path):
    state = {
        "input_data": {},
        "breakpoint": {"component": "comp1", "visits": 0},
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
        Pipeline.load_state(state_file)
