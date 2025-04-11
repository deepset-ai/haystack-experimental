# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import json
import pytest

from haystack.components.joiners import BranchJoiner
from haystack.core.component import component
from haystack.dataclasses import ChatMessage, GeneratedAnswer
from haystack_experimental.core.errors import PipelineRuntimeError
from haystack_experimental.core.pipeline.pipeline import Pipeline


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

    def test__run_component_success(self):
        """Test successful component execution"""
        joiner_1 = BranchJoiner(type_=str)
        joiner_2 = BranchJoiner(type_=str)
        pp = Pipeline()
        pp.add_component("joiner_1", joiner_1)
        pp.add_component("joiner_2", joiner_2)
        pp.connect("joiner_1", "joiner_2")
        inputs = {"joiner_1": {"value": [{"sender": None, "value": "test_value"}]}}

        outputs = pp._run_component(
            component=pp._get_component_with_graph_metadata_and_visits("joiner_1", 0),
            inputs=inputs,
            component_visits={"joiner_1": 0, "joiner_2": 0},
        )

        assert outputs == {"value": "test_value"}
        # We remove input in greedy variadic sockets, even if they are from the user
        assert "value" not in inputs["joiner_1"]

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

        inputs = {"wrong": {"value": [{"sender": None, "value": "test_value"}]}}

        with pytest.raises(PipelineRuntimeError) as exc_info:
            pp._run_component(
                component=pp._get_component_with_graph_metadata_and_visits("wrong", 0),
                inputs=inputs,
                component_visits={"wrong": 0},
            )

        assert "didn't return a dictionary" in str(exc_info.value)

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

    def test_transform_json_structure_unwraps_sender_value(self):        
        data = {
            "key1": [{"sender": None, "value": "some value"}],
            "key2": [{"sender": "comp1", "value": 42}],
            "key3": "direct value"
        }
        
        result = Pipeline.transform_json_structure(data)
        
        assert result == {
            "key1": "some value",
            "key2": 42,
            "key3": "direct value"
        }
        
    def test_transform_json_structure_handles_nested_structures(self):        
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
        
        result = Pipeline.transform_json_structure(data)
        
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
        
    def test_serialize_component_input_handles_objects_with_to_dict(self):        
        class TestObject:
            def __init__(self, value):
                self.value = value
                
            def to_dict(self):
                return {"value": self.value}
                
        obj = TestObject("test")    
        result = Pipeline._serialize_component_input(obj)        
        assert result == {
            "_type": "TestObject",
            "value": "test"
        }
        
    def test_serialize_component_input_handles_objects_without_to_dict(self):        
        class TestObject:
            def __init__(self, value):
                self.value = value
                
        obj = TestObject("test")        
        result = Pipeline._serialize_component_input(obj)        
        assert result == {
            "_type": "TestObject",
            "attributes": {"value": "test"}
        }
        
    def test_serialize_component_input_handles_nested_structures(self):        
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
        result = Pipeline._serialize_component_input(data)
        
        assert result["key1"]["_type"] == "TestObject"
        assert result["key2"][0]["_type"] == "TestObject"
        assert result["key2"][1] == "string"
        assert result["key3"]["nested"]["_type"] == "TestObject"
        
    def test_deserialize_component_input_handles_primitive_types(self):        
        data = {
            "string": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None
        }        
        result = Pipeline._deserialize_component_input(data)        
        assert result == data
        
    def test_deserialize_component_input_handles_lists(self):        
        data = {
            "primitive_list": [1, 2, 3],
            "mixed_list": [1, "string", True]
        }        
        result = Pipeline._deserialize_component_input(data)        
        assert result == data
        
    def test_deserialize_component_input_handles_dicts(self):
        data = {
            "key1": "value1",
            "key2": {"nested": "value2"}
        }        
        result = Pipeline._deserialize_component_input(data)        
        assert result == data
        
    def test_validate_resume_state_validates_required_keys(self):        
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
            
    def test_validate_resume_state_validates_component_consistency(self):        
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
            
    def test_validate_resume_state_validates_valid_state(self):        
        state = {
            "input_data": {},
            "breakpoint": {"component": "comp1", "visits": 0},
            "pipeline_state": {
                "inputs": {},
                "component_visits": {"comp1": 0, "comp2": 0},
                "ordered_component_names": ["comp1", "comp2"]
            }
        }
        
        assert Pipeline._validate_resume_state(state)
        
    def test_load_state_loads_valid_state(self, tmp_path):        
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

    def test_load_state_handles_invalid_state(self, tmp_path):        
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
