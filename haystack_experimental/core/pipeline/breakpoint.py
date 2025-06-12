# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-return-statements, too-many-positional-arguments

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from haystack import Answer, Document, ExtractedAnswer, logging
from haystack.dataclasses import ChatMessage, SparseEmbedding

from haystack_experimental.core.errors import PipelineInvalidResumeStateError
from haystack_experimental.dataclasses import GeneratedAnswer

logger = logging.getLogger(__name__)


def _validate_breakpoint(pipeline_breakpoint: Tuple[str, Optional[int]], graph: Any) -> Tuple[str, int]:
    """
    Validates the pipeline_breakpoint passed to the pipeline.

    Makes sure the breakpoint contains a valid components registered in the pipeline.
    If the visit is not given, it is assumed to be 0, it will break on the first visit.

    :param pipeline_breakpoint: Tuple of component name and visit count at which the pipeline should stop.
    :returns:
        Tuple of valid pipeline_breakpoint.
    """

    if pipeline_breakpoint and pipeline_breakpoint[0] not in graph.nodes:
        raise ValueError(f"pipeline_breakpoint {pipeline_breakpoint} is not a registered component in the pipeline")
    valid_breakpoint: Tuple[str, int] = (
        (pipeline_breakpoint[0], 0 if pipeline_breakpoint[1] is None else pipeline_breakpoint[1])
        if pipeline_breakpoint
        else None
    )
    return valid_breakpoint


def _validate_pipeline_state(resume_state: Dict[str, Any], graph: Any) -> None:
    """
    Validates that the resume_state contains valid configuration for the current pipeline.

    Raises a PipelineRuntimeError if any component is missing or if the state structure is invalid.

    :param resume_state: The saved state to validate.
    """

    pipeline_state = resume_state["pipeline_state"]
    valid_components = set(graph.nodes.keys())

    # Check if the ordered_component_names are valid components in the pipeline
    missing_ordered = set(pipeline_state["ordered_component_names"]) - valid_components
    if missing_ordered:
        raise PipelineInvalidResumeStateError(
            f"Invalid resume state: components {missing_ordered} in 'ordered_component_names' "
            f"are not part of the current pipeline."
        )

    # Check if the input_data is valid components in the pipeline
    missing_input = set(resume_state["input_data"].keys()) - valid_components
    if missing_input:
        raise PipelineInvalidResumeStateError(
            f"Invalid resume state: components {missing_input} in 'input_data' are not part of the current pipeline."
        )

    # Validate 'component_visits'
    missing_visits = set(pipeline_state["component_visits"].keys()) - valid_components
    if missing_visits:
        raise PipelineInvalidResumeStateError(
            f"Invalid resume state: components {missing_visits} in 'component_visits' "
            f"are not part of the current pipeline."
        )

    logger.info(
        f"Resuming pipeline from component: {resume_state['pipeline_breakpoint']['component']} "
        f"(visit {resume_state['pipeline_breakpoint']['visits']})"
    )


def _validate_resume_state(state: Dict[str, Any]) -> None:
    """
    Validates the loaded pipeline state.

    Ensures that the state contains required keys: "input_data", "pipeline_breakpoint", and "pipeline_state".

    Raises:
        ValueError: If required keys are missing or the component sets are inconsistent.
    """

    # top-level state has all required keys
    required_top_keys = {"input_data", "pipeline_breakpoint", "pipeline_state"}
    missing_top = required_top_keys - state.keys()
    if missing_top:
        raise ValueError(f"Invalid state file: missing required keys {missing_top}")

    # pipeline_state has the necessary keys
    pipeline_state = state["pipeline_state"]
    required_pipeline_keys = {"inputs", "component_visits", "ordered_component_names"}
    missing_pipeline = required_pipeline_keys - pipeline_state.keys()
    if missing_pipeline:
        raise ValueError(f"Invalid pipeline_state: missing required keys {missing_pipeline}")

    # component_visits and ordered_component_names must be consistent
    components_in_state = set(pipeline_state["component_visits"].keys())
    components_in_order = set(pipeline_state["ordered_component_names"])

    if components_in_state != components_in_order:
        raise ValueError(
            f"Inconsistent state: components in pipeline_state['component_visits'] {components_in_state} "
            f"do not match components in ordered_component_names {components_in_order}"
        )

    logger.info("Passed resume state validated successfully.")


def load_state(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a saved pipeline state.

    :param file_path: Path to the state file
    :returns:
        Dict containing the loaded state
    """

    file_path = Path(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON file {file_path}: {str(e)}", e.doc, e.pos)
    except IOError as e:
        raise IOError(f"Error reading {file_path}: {str(e)}")

    try:
        _validate_resume_state(state=state)
    except ValueError as e:
        raise ValueError(f"Invalid pipeline state from {file_path}: {str(e)}")

    logger.info(f"Successfully loaded pipeline state from: {file_path}")
    return state


def _save_state(
    inputs: Dict[str, Any],
    component_name: str,
    component_visits: Dict[str, int],
    callback_fun: Optional[Callable[..., Any]] = None,
    debug_path: Optional[Union[str, Path]] = None,
    original_input_data: Optional[Dict[str, Any]] = None,
    ordered_component_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Save the pipeline state to a file.

    :param inputs: The current pipeline state inputs.
    :param component_name: The name of the component that triggered the breakpoint.
    :param component_visits: The visit count of the component that triggered the breakpoint.
    :param callback_fun: A function to call with the saved state.
    :param debug_path: The path to save the state to.
    :param original_input_data: The original input data.
    :param ordered_component_names: The ordered component names.
    :raises:
        Exception: If the debug_path is not a string or a Path object, or if saving the JSON state fails.

    :returns:
        The saved state dictionary
    """
    dt = datetime.now()

    state = {
        "input_data": _serialize_component_input(original_input_data),  # original input data
        "timestamp": dt.isoformat(),
        "pipeline_breakpoint": {"component": component_name, "visits": component_visits[component_name]},
        "pipeline_state": {
            "inputs": _serialize_component_input(inputs),  # current pipeline state inputs
            "component_visits": component_visits,
            "ordered_component_names": ordered_component_names,
        },
    }

    if not debug_path:
        return state

    debug_path = Path(debug_path) if isinstance(debug_path, str) else debug_path
    if not isinstance(debug_path, Path):
        raise ValueError("Debug path must be a string or a Path object.")
    debug_path.mkdir(exist_ok=True)
    file_name = Path(f"{component_name}_{dt.strftime('%Y_%m_%d_%H_%M_%S')}.json")

    try:
        with open(debug_path / file_name, "w") as f_out:
            json.dump(state, f_out, indent=2)
        logger.info(f"Pipeline state saved at: {file_name}")

        # pass the state to some user-defined callback function
        if callback_fun is not None:
            callback_fun(state)

        return state

    except Exception as e:
        logger.error(f"Failed to save pipeline state: {str(e)}")
        raise


def _deserialize_component_input(value: Any) -> Any:  # noqa: PLR0911
    """
    Tries to deserialize any type of input that can be passed to as input to a pipeline component.

    For primitive values, it returns the value as is, but for complex types, it tries to deserialize them.
    """

    # None or primitive types are returned as is
    if not value or isinstance(value, (str, int, float, bool)):
        return value

    # list of primitive types are returned as is
    if isinstance(value, list) and all(isinstance(i, (str, int, float, bool)) for i in value):
        return value

    if isinstance(value, list):
        # list of lists are called recursively
        if all(isinstance(i, list) for i in value):
            return [_deserialize_component_input(i) for i in value]
        # list of dicts are called recursively
        if all(isinstance(i, dict) for i in value):
            return [_deserialize_component_input(i) for i in value]

    # Define the mapping of types to their deserialization functions
    _type_deserializers = {
        "Answer": Answer.from_dict,
        "ChatMessage": ChatMessage.from_dict,
        "Document": Document.from_dict,
        "ExtractedAnswer": ExtractedAnswer.from_dict,
        "GeneratedAnswer": GeneratedAnswer.from_dict,
        "SparseEmbedding": SparseEmbedding.from_dict,
    }

    # check if the dictionary has a "_type" key and if it's a known type
    if isinstance(value, dict):
        if "_type" in value:
            type_name = value.pop("_type")
            if type_name in _type_deserializers:
                return _type_deserializers[type_name](value)

        # If not a known type, recursively deserialize each item in the dictionary
        return {k: _deserialize_component_input(v) for k, v in value.items()}

    return value


def _transform_json_structure(data: Union[Dict[str, Any], List[Any], Any]) -> Any:
    """
    Transforms a JSON structure by removing the 'sender' key and moving the 'value' to the top level.

    For example:
    "key": [{"sender": null, "value": "some value"}] -> "key": "some value"

    :param data: The JSON structure to transform.
    :returns: The transformed structure.
    """
    if isinstance(data, dict):
        # If this dict has both 'sender' and 'value', return just the value
        if "value" in data and "sender" in data:
            return data["value"]
        # Otherwise, recursively process each key-value pair
        return {k: _transform_json_structure(v) for k, v in data.items()}

    if isinstance(data, list):
        # First, transform each item in the list.
        transformed = [_transform_json_structure(item) for item in data]
        # If the original list has exactly one element and that element was a dict
        # with 'sender' and 'value', then unwrap the list.
        if len(data) == 1 and isinstance(data[0], dict) and "value" in data[0] and "sender" in data[0]:
            return transformed[0]
        return transformed

    # For other data types, just return the value as is.
    return data


def _serialize_component_input(value: Any) -> Any:
    """
    Serializes, so it can be saved to a file, any type of input to a pipeline component.

    :param value: The value to serialize.
    :returns: The serialized value that can be saved to a file.
    """
    value = _transform_json_structure(value)
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        serialized_value = value.to_dict()
        serialized_value["_type"] = value.__class__.__name__
        return serialized_value

    # this is a hack to serialize inputs that don't have a to_dict
    elif hasattr(value, "__dict__"):
        return {
            "_type": value.__class__.__name__,
            "attributes": value.__dict__,
        }

    # recursively serialize all inputs in a dict
    elif isinstance(value, dict):
        return {k: _serialize_component_input(v) for k, v in value.items()}

    # recursively serialize all inputs in lists or tuples
    elif isinstance(value, list):
        return [_serialize_component_input(item) for item in value]

    return value
