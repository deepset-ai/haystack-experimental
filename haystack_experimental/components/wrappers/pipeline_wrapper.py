"""Module providing a wrapper for Haystack pipelines with input/output mapping capabilities."""
from typing import Any, Dict, List, Optional, Tuple
from haystack import component, Pipeline


class InvalidMappingError(Exception):
    """Raised when input or output mappings are invalid or type conflicts are found."""
    pass


@component
class PipelineWrapper:
    """A wrapper component that encapsulates a Haystack pipeline with flexible I/O mapping.

    This wrapper allows for remapping of input and output socket names between the wrapped
    pipeline and the external interface. It handles type checking and verification of all
    mappings.

    :param pipeline: The pipeline wrapped by the component
    :param input_mapping: Mapping from wrapper input names to lists of pipeline socket paths
        in format "component_name.socket_name"
    :param output_mapping: Mapping from pipeline socket paths to wrapper output names
    :raises InvalidMappingError: If any input or output mappings are invalid or if type
        conflicts are detected
    """

    def __init__(
            self,
            pipeline: Pipeline,
            input_mapping: Optional[Dict[str, List[str]]] = None,
            output_mapping: Optional[Dict[str, str]] = None
    ):
        """Initialize the pipeline wrapper with optional I/O mappings.

        :param pipeline: The pipeline to wrap
        :param input_mapping: Optional input name mapping configuration
        :param output_mapping: Optional output name mapping configuration
        """
        self.pipeline = pipeline
        self.input_mapping = input_mapping or {}
        self.output_mapping = output_mapping or {}
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize pipeline by resolving input/output types and setting them on wrapper."""
        self._resolve_and_set_input_types()
        output_types = self._resolve_output_types()
        component.set_output_types(self, **output_types)

    @staticmethod
    def _split_component_path(path: str) -> Tuple[str, str]:
        """Split a component path into component name and socket name.

        :param path: String in format "component_name.socket_name"
        :returns: Tuple of (component_name, socket_name)
        :raises InvalidMappingError: If path format is invalid
        """
        try:
            comp_name, socket_name = path.split(".")
            return comp_name, socket_name
        except ValueError:
            raise InvalidMappingError(
                f"Invalid path format: {path}. Expected 'component_name.socket_name'"
            )

    def _resolve_and_set_input_types(self) -> None:
        """Resolve input types from pipeline and set them on wrapper.

        Examines the pipeline's declared inputs and merges them with the input mapping
        to check for type conflicts and set appropriate input types on the wrapper component.

        :raises InvalidMappingError: If type conflicts are found or mapped components/sockets
            don't exist
        """
        pipeline_inputs = self.pipeline.inputs(include_components_with_connected_inputs=False)

        if self.input_mapping:
            self._handle_explicit_input_mapping(pipeline_inputs)
        else:
            self._handle_auto_input_mapping(pipeline_inputs)

    def _handle_explicit_input_mapping(self, pipeline_inputs: Dict[str, Dict[str, Any]]) -> None:
        """Handle case where explicit input mapping is provided.

        :param pipeline_inputs: Dictionary of pipeline input specifications
        :raises InvalidMappingError: If mapping is invalid or type conflicts exist
        """
        for wrapper_input_name, pipeline_input_paths in self.input_mapping.items():
            if not isinstance(pipeline_input_paths, list):
                pipeline_input_paths = [pipeline_input_paths]

            resolved_type = None
            resolved_default = None

            for path in pipeline_input_paths:
                comp_name, socket_name = self._split_component_path(path)

                if comp_name not in pipeline_inputs:
                    raise InvalidMappingError(f"Component '{comp_name}' not found in pipeline inputs.")

                if socket_name not in pipeline_inputs[comp_name]:
                    raise InvalidMappingError(
                        f"Input socket '{socket_name}' not found in component '{comp_name}'."
                    )

                socket_info = pipeline_inputs[comp_name][socket_name]
                current_type = socket_info["type"]

                if resolved_type is None:
                    resolved_type = current_type
                elif current_type != resolved_type:
                    raise InvalidMappingError(
                        f"Type conflict for wrapper input '{wrapper_input_name}': "
                        f"found both {resolved_type} and {current_type}"
                    )

                if not socket_info["is_mandatory"]:
                    current_default = socket_info.get("default_value")
                    if resolved_default is None:
                        resolved_default = current_default
                    elif current_default != resolved_default:
                        raise InvalidMappingError(
                            f"Default value conflict for input '{wrapper_input_name}': "
                            f"found {resolved_default} and {current_default}"
                        )

            if resolved_default is not None:
                component.set_input_type(self, wrapper_input_name, resolved_type, default=resolved_default)
            else:
                component.set_input_type(self, wrapper_input_name, resolved_type)

    def _handle_auto_input_mapping(self, pipeline_inputs: Dict[str, Dict[str, Any]]) -> None:
        """Handle case where input mapping should be auto-detected.

        :param pipeline_inputs: Dictionary of pipeline input specifications
        :raises InvalidMappingError: If type conflicts exist between components
        """
        aggregated_inputs = {}

        for comp_name, inputs_dict in pipeline_inputs.items():
            for socket_name, socket_info in inputs_dict.items():
                if socket_name not in aggregated_inputs:
                    aggregated_inputs[socket_name] = socket_info
                else:
                    existing_type = aggregated_inputs[socket_name]["type"]
                    if existing_type != socket_info["type"]:
                        raise InvalidMappingError(
                            f"Type conflict for input '{socket_name}': "
                            f"found {existing_type} and {socket_info['type']}"
                        )

        for input_name, info in aggregated_inputs.items():
            if info["is_mandatory"]:
                component.set_input_type(self, input_name, info["type"])
            else:
                default_val = info.get("default_value")
                component.set_input_type(
                    self, input_name, info["type"],
                    default=default_val if default_val is not None else None
                )

    def _resolve_output_types(self) -> Dict[str, Any]:
        """Resolve output types based on pipeline outputs and mapping.

        :returns: Dictionary mapping wrapper output names to their types
        :raises InvalidMappingError: If mapping is invalid or output conflicts exist
        """
        pipeline_outputs = self.pipeline.outputs(include_components_with_connected_outputs=False)
        resolved_outputs: Dict[str, Any] = {}
        used_output_names = set()

        if self.output_mapping:
            return self._handle_explicit_output_mapping(pipeline_outputs)
        else:
            return self._handle_auto_output_mapping(pipeline_outputs)

    def _handle_explicit_output_mapping(self, pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Handle case where explicit output mapping is provided.

        :param pipeline_outputs: Dictionary of pipeline output specifications
        :returns: Dictionary of resolved output types
        :raises InvalidMappingError: If mapping is invalid
        """
        resolved_outputs = {}
        for pipeline_output_path, wrapper_output_name in self.output_mapping.items():
            comp_name, socket_name = self._split_component_path(pipeline_output_path)

            if comp_name not in pipeline_outputs:
                raise InvalidMappingError(f"Component '{comp_name}' not found among pipeline outputs.")
            if socket_name not in pipeline_outputs[comp_name]:
                raise InvalidMappingError(
                    f"Output socket '{socket_name}' not found in component '{comp_name}'."
                )
            if wrapper_output_name in resolved_outputs:
                raise InvalidMappingError(
                    f"Duplicate wrapper output name '{wrapper_output_name}' in output_mapping."
                )

            resolved_outputs[wrapper_output_name] = pipeline_outputs[comp_name][socket_name]["type"]

        return resolved_outputs

    def _handle_auto_output_mapping(self, pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Handle case where output mapping should be auto-detected.

        :param pipeline_outputs: Dictionary of pipeline output specifications
        :returns: Dictionary of resolved output types
        :raises InvalidMappingError: If output conflicts exist
        """
        resolved_outputs = {}
        used_output_names = set()

        for comp_name, outputs_dict in pipeline_outputs.items():
            for socket_name, socket_info in outputs_dict.items():
                if socket_name in used_output_names:
                    raise InvalidMappingError(
                        f"Output name conflict: '{socket_name}' is produced by multiple components. "
                        "Please provide an output_mapping to disambiguate."
                    )
                resolved_outputs[socket_name] = socket_info["type"]
                used_output_names.add(socket_name)

        return resolved_outputs

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """Run the wrapped pipeline with the given inputs.

        This method:
        1. Maps input kwargs to pipeline component inputs
        2. Executes the pipeline
        3. Maps pipeline outputs back to wrapper outputs

        :param kwargs: Keyword arguments matching wrapper input names
        :returns: Dictionary mapping wrapper output names to values
        :raises ValueError: If no pipeline is configured
        :raises InvalidMappingError: If output conflicts occur during auto-mapping
        """
        if not self.pipeline:
            raise ValueError("No pipeline configured. Provide a valid pipeline before calling run().")

        pipeline_inputs = self._prepare_pipeline_inputs(kwargs)
        pipeline_outputs = self.pipeline.run(**pipeline_inputs)
        return self._process_pipeline_outputs(pipeline_outputs)

    def _prepare_pipeline_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Prepare inputs for pipeline execution.

        :param kwargs: Input arguments provided to wrapper
        :returns: Dictionary of pipeline inputs in required format
        """
        if self.input_mapping:
            return self._map_explicit_inputs(kwargs)
        else:
            return self._map_auto_inputs(kwargs)

    def _map_explicit_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Map inputs according to explicit input mapping.

        :param kwargs: Input arguments provided to wrapper
        :returns: Dictionary of mapped pipeline inputs
        """
        pipeline_inputs = {}
        for wrapper_input_name, pipeline_input_paths in self.input_mapping.items():
            if wrapper_input_name not in kwargs:
                continue

            if not isinstance(pipeline_input_paths, list):
                pipeline_input_paths = [pipeline_input_paths]

            for socket_path in pipeline_input_paths:
                comp_name, input_name = self._split_component_path(socket_path)
                if comp_name not in pipeline_inputs:
                    pipeline_inputs[comp_name] = {}
                pipeline_inputs[comp_name][input_name] = kwargs[wrapper_input_name]

        return pipeline_inputs

    def _map_auto_inputs(self, kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Map inputs automatically based on matching names.

        :param kwargs: Input arguments provided to wrapper
        :returns: Dictionary of mapped pipeline inputs
        """
        pipeline_inputs = {}
        pipeline_decl_inputs = self.pipeline.inputs(include_components_with_connected_inputs=False)

        for comp_name, inputs_dict in pipeline_decl_inputs.items():
            for socket_name in inputs_dict.keys():
                if socket_name in kwargs:
                    pipeline_inputs.setdefault(comp_name, {})[socket_name] = kwargs[socket_name]

        return pipeline_inputs

    def _process_pipeline_outputs(self, pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process outputs from pipeline execution.

        :param pipeline_outputs: Raw outputs from pipeline execution
        :returns: Dictionary of processed outputs
        :raises InvalidMappingError: If output conflicts occur during auto-mapping
        """
        if self.output_mapping:
            return self._map_explicit_outputs(pipeline_outputs)
        else:
            return self._map_auto_outputs(pipeline_outputs)

    def _map_explicit_outputs(self, pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Map outputs according to explicit output mapping.

        :param pipeline_outputs: Raw outputs from pipeline execution
        :returns: Dictionary of mapped outputs
        """
        outputs = {}
        for pipeline_output_path, wrapper_output_name in self.output_mapping.items():
            comp_name, socket_name = self._split_component_path(pipeline_output_path)
            if comp_name in pipeline_outputs and socket_name in pipeline_outputs[comp_name]:
                outputs[wrapper_output_name] = pipeline_outputs[comp_name][socket_name]
        return outputs

    def _map_auto_outputs(self, pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Map outputs automatically based on matching names.

        :param pipeline_outputs: Raw outputs from pipeline execution
        :returns: Dictionary of mapped outputs
        :raises InvalidMappingError: If output conflicts occur
        """
        outputs = {}
        seen = set()

        for comp_name, comp_output_dict in pipeline_outputs.items():
            for socket_name, value in comp_output_dict.items():
                if socket_name in seen:
                    raise InvalidMappingError(
                        f"Output conflict: output '{socket_name}' is produced by multiple components. "
                        "Provide an output_mapping to disambiguate."
                    )
                outputs[socket_name] = value
                seen.add(socket_name)

        return outputs