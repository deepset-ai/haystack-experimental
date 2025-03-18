# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple

from haystack import Pipeline, component
from haystack.core.pipeline.utils import parse_connect_string
from haystack.core.serialization import default_from_dict, default_to_dict, generate_qualified_class_name

from haystack_experimental.core.super_component.utils import _delegate_default, is_compatible


class InvalidMappingError(Exception):
    """Raised when input or output mappings are invalid or type conflicts are found."""
    pass


@component
class SuperComponent:
    """
        A class for creating super components that wrap around a Pipeline.

        This component allows for remapping of input and output socket names between the wrapped
        pipeline and the external interface. It handles type checking and verification of all
        mappings.

        :param pipeline: The pipeline wrapped by the component
        :param input_mapping: Mapping from component input names to lists of pipeline socket paths
            in format "component_name.socket_name"
        :param output_mapping: Mapping from pipeline socket paths to component output names
        :raises InvalidMappingError: If any input or output mappings are invalid or if type
            conflicts are detected
        :raises ValueError: If no pipeline is provided
        """

    def __init__(
        self,
        pipeline: Pipeline,
        input_mapping: Optional[Dict[str, List[str]]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize the component with optional I/O mappings.

        :param pipeline: The pipeline to wrap
        :param input_mapping: Optional input name mapping configuration
        :param output_mapping: Optional output name mapping configuration
        """
        if pipeline is None:
            raise ValueError("Pipeline must be provided to SuperComponent.")

        self.pipeline: Pipeline = pipeline
        self._warmed_up = False

        # Determine input types based on pipeline and mapping
        pipeline_inputs = self.pipeline.inputs()
        if input_mapping is None:
            input_types, auto_input_mapping = self._handle_auto_input_mapping(pipeline_inputs)
            resolved_input_mapping = auto_input_mapping
        else:
            input_types = self._handle_explicit_input_mapping(pipeline_inputs, input_mapping)
            resolved_input_mapping = input_mapping

        # Set input types on the component
        for input_name, info in input_types.items():
            if info["is_mandatory"]:
                component.set_input_type(self, input_name, info["type"])
            else:
                component.set_input_type(self, input_name, info["type"], default=_delegate_default)

        self.input_mapping: Dict[str, List[str]] = resolved_input_mapping
        self._original_input_mapping = input_mapping

        # Set output types based on pipeline and mapping
        pipeline_outputs = self.pipeline.outputs()
        if output_mapping is None:
            output_types, auto_output_mapping = self._handle_auto_output_mapping(pipeline_outputs)
            resolved_output_mapping = auto_output_mapping
        else:
            output_types = self._handle_explicit_output_mapping(pipeline_outputs, output_mapping)
            resolved_output_mapping = output_mapping

        # Set output types on the component
        component.set_output_types(self, **output_types)

        self.output_mapping: Dict[str, str] = resolved_output_mapping
        self._original_output_mapping = output_mapping

    def warm_up(self) -> None:
        """
        Warms up the pipeline if it has not been warmed up before.
        """
        if not self._warmed_up:
            self.pipeline.warm_up()
            self._warmed_up = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the SuperComponent to a dictionary representation.

        Must be overwritten for custom component implementations that inherit from SuperComponent.

        :return: Dictionary containing serialized super component data
        """
        return self._to_super_component_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuperComponent":
        """
        Create the SuperComponent instance from a dictionary representation.

        Must be overwritten for custom component implementations that inherit from SuperComponent.

        :param data: Dictionary containing serialized super component data
        :return: New PipelineWrapper instance
        """
        pipeline = Pipeline.from_dict(data["init_parameters"]["pipeline"])
        data["init_parameters"]["pipeline"] = pipeline
        return default_from_dict(cls, data)

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Run the wrapped pipeline with the given inputs.

        This method:
        1. Maps input kwargs to pipeline component inputs
        2. Executes the pipeline
        3. Maps pipeline outputs back to wrapper outputs

        :param kwargs: Keyword arguments matching wrapper input names
        :return: Dictionary mapping wrapper output names to values
        :raises ValueError: If no pipeline is configured
        :raises InvalidMappingError: If output conflicts occur during auto-mapping
        """
        if self._warmed_up is False:
            raise RuntimeError("SuperComponent wasn't warmed up. Run 'warm_up()' before calling 'run()'.")

        filtered_inputs = {param: value for param, value in kwargs.items() if value != _delegate_default}
        pipeline_inputs = self._map_explicit_inputs(input_mapping=self.input_mapping, inputs=filtered_inputs)
        pipeline_outputs = self.pipeline.run(data=pipeline_inputs)
        return self._map_explicit_outputs(pipeline_outputs, self.output_mapping)

    @staticmethod
    def _split_component_path(path: str) -> Tuple[str, Optional[str]]:
        """
        Split a component path into component name and socket name.

        :param path: String in format "component_name.socket_name"
        :return: Tuple of (component_name, socket_name)
        :raises InvalidMappingError: If path format is invalid
        """
        comp_name, socket_name = parse_connect_string(path)
        if socket_name is None:
            raise InvalidMappingError(f"Invalid path format: '{path}'. Expected 'component_name.socket_name'.")
        return comp_name, socket_name

    def _handle_explicit_input_mapping(  # noqa: PLR0912
        self, pipeline_inputs: Dict[str, Dict[str, Any]], input_mapping: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Handle case where explicit input mapping is provided.

        :param pipeline_inputs: Dictionary of pipeline input specifications
        :param input_mapping: Mapping from wrapper inputs to pipeline socket paths
        :returns: The resolved input type data resolved according to the input mapping.
        :raises InvalidMappingError: If mapping is invalid or type conflicts exist
        """
        aggregated_inputs: Dict[str, Dict[str, Any]] = {}
        for wrapper_input_name, pipeline_input_paths in input_mapping.items():

            if not isinstance(pipeline_input_paths, list):
                raise InvalidMappingError(f"Input paths for '{wrapper_input_name}' must be a list of strings.")

            for path in pipeline_input_paths:
                comp_name, socket_name = self._split_component_path(path)

                if comp_name not in pipeline_inputs:
                    raise InvalidMappingError(f"Component '{comp_name}' not found in pipeline inputs.")

                if socket_name not in pipeline_inputs[comp_name]:
                    raise InvalidMappingError(f"Input socket '{socket_name}' not found in component '{comp_name}'.")

                socket_info = pipeline_inputs[comp_name][socket_name]
                if existing_socket_info := aggregated_inputs.get(wrapper_input_name):
                    # TODO is_compatible should determine least common denominator of type overlap
                    #      and set that as the type for the wrapper input
                    if not is_compatible(existing_socket_info["type"], socket_info["type"]):
                        raise InvalidMappingError(
                            f"Type conflict for input '{socket_name}' from component '{comp_name}'. "
                            f"We already have type {existing_socket_info['type']} for '{socket_name}' which is not"
                            f"compatible with {socket_info['type']}."
                        )

                    # If any socket requires mandatory inputs then pass it to the wrapper and use the type from the
                    # mandatory socket
                    if not aggregated_inputs[wrapper_input_name]["is_mandatory"]:
                        aggregated_inputs[wrapper_input_name]["is_mandatory"] = socket_info["is_mandatory"]
                        aggregated_inputs[wrapper_input_name]["type"] = socket_info["type"]
                else:
                    aggregated_inputs[wrapper_input_name] = socket_info

        return aggregated_inputs

    @staticmethod
    def _handle_auto_input_mapping(
        pipeline_inputs: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
        """
        Handle case where input mapping should be auto-detected.

        :param pipeline_inputs: Dictionary of pipeline input specifications
        :returns: The resolved input type data resolved according to auto-resolution.
        :raises InvalidMappingError: If type conflicts exist between components
        """
        aggregated_inputs: Dict[str, Dict[str, Any]] = {}
        input_mapping: Dict[str, List[str]] = {}

        for comp_name, inputs_dict in pipeline_inputs.items():
            for socket_name, socket_info in inputs_dict.items():
                if existing_socket_info := aggregated_inputs.get(socket_name):
                    # TODO is_compatible should determine least common denominator of type overlap
                    #      and set that as the type for the wrapper input
                    if not is_compatible(existing_socket_info["type"], socket_info["type"]):
                        raise InvalidMappingError(
                            f"Type conflict for input '{socket_name}' from component '{comp_name}'. "
                            f"We already have type {existing_socket_info['type']} for '{socket_name}' which is not"
                            f"compatible with {socket_info['type']}."
                        )

                    # If any socket requires mandatory inputs then pass it to the wrapper and use the type from the
                    # mandatory socket
                    if not existing_socket_info["is_mandatory"]:
                        aggregated_inputs[socket_name]["is_mandatory"] = socket_info["is_mandatory"]
                        aggregated_inputs[socket_name]["type"] = socket_info["type"]

                    # Add the component name to the input mapping
                    input_mapping[socket_name].append(f"{comp_name}.{socket_name}")
                else:
                    aggregated_inputs[socket_name] = socket_info
                    input_mapping[socket_name] = [f"{comp_name}.{socket_name}"]

        return aggregated_inputs, input_mapping

    def _handle_explicit_output_mapping(
        self, pipeline_outputs: Dict[str, Dict[str, Any]], output_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Handle case where explicit output mapping is provided.

        :param pipeline_outputs: Dictionary of pipeline output specifications
        :param output_mapping: Mapping from pipeline paths to wrapper output names
        :return: Dictionary of resolved output types
        :raises InvalidMappingError: If mapping is invalid
        """
        resolved_outputs = {}
        for pipeline_output_path, wrapper_output_name in output_mapping.items():
            if not isinstance(wrapper_output_name, str):
                raise InvalidMappingError("Output names in output_mapping must be strings.")

            comp_name, socket_name = self._split_component_path(pipeline_output_path)

            if comp_name not in pipeline_outputs:
                raise InvalidMappingError(f"Component '{comp_name}' not found among pipeline outputs.")
            if socket_name not in pipeline_outputs[comp_name]:
                raise InvalidMappingError(f"Output socket '{socket_name}' not found in component '{comp_name}'.")
            if wrapper_output_name in resolved_outputs:
                raise InvalidMappingError(f"Duplicate output name '{wrapper_output_name}' in output_mapping.")

            resolved_outputs[wrapper_output_name] = pipeline_outputs[comp_name][socket_name]["type"]

        return resolved_outputs

    @staticmethod
    def _handle_auto_output_mapping(
        pipeline_outputs: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Handle case where output mapping should be auto-detected.

        :param pipeline_outputs: Dictionary of pipeline output specifications
        :return: Dictionary of resolved output types
        :raises InvalidMappingError: If output conflicts exist
        """
        resolved_outputs = {}
        output_mapping = {}
        used_output_names: set[str] = set()

        for outputs_dict in pipeline_outputs.values():
            for socket_name, socket_info in outputs_dict.items():
                if socket_name in used_output_names:
                    raise InvalidMappingError(
                        f"Output name conflict: '{socket_name}' is produced by multiple components. "
                        "Please provide an output_mapping to resolve this conflict."
                    )
                resolved_outputs[socket_name] = socket_info["type"]
                used_output_names.add(socket_name)
                output_mapping[socket_name] = socket_name

        return resolved_outputs, output_mapping

    def _map_explicit_inputs(
        self, input_mapping: Dict[str, List[str]], inputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Map inputs according to explicit input mapping.

        :param input_mapping: Mapping configuration for inputs
        :param inputs: Input arguments provided to wrapper
        :return: Dictionary of mapped pipeline inputs
        """
        pipeline_inputs: Dict[str, Dict[str, Any]] = {}
        for wrapper_input_name, pipeline_input_paths in input_mapping.items():
            if wrapper_input_name not in inputs:
                continue

            for socket_path in pipeline_input_paths:
                comp_name, input_name = self._split_component_path(socket_path)
                if comp_name not in pipeline_inputs:
                    pipeline_inputs[comp_name] = {}
                pipeline_inputs[comp_name][input_name] = inputs[wrapper_input_name]

        return pipeline_inputs

    def _map_explicit_outputs(
        self, pipeline_outputs: Dict[str, Dict[str, Any]], output_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Map outputs according to explicit output mapping.

        :param pipeline_outputs: Raw outputs from pipeline execution
        :param output_mapping: Output mapping configuration
        :return: Dictionary of mapped outputs
        """
        outputs: Dict[str, Any] = {}
        for pipeline_output_path, wrapper_output_name in output_mapping.items():
            comp_name, socket_name = self._split_component_path(pipeline_output_path)
            if comp_name in pipeline_outputs and socket_name in pipeline_outputs[comp_name]:
                outputs[wrapper_output_name] = pipeline_outputs[comp_name][socket_name]
        return outputs

    def _to_super_component_dict(self) -> Dict[str, Any]:
        """
        Convert to a SuperComponent dictionary representation.

        :return: Dictionary containing serialized SuperComponent data
        """
        serialized_pipeline = self.pipeline.to_dict()
        serialized = default_to_dict(
            self,
            pipeline=serialized_pipeline,
            input_mapping=self._original_input_mapping,
            output_mapping=self._original_output_mapping
        )
        serialized["type"] = generate_qualified_class_name(SuperComponent)
        return serialized
