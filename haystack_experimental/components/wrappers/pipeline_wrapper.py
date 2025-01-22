from typing import Any, Dict, List, Optional, Tuple, Type, Union, get_args, get_origin

from haystack import Pipeline, component
from haystack.core.component.types import GreedyVariadic, Variadic
from haystack.core.serialization import default_from_dict, default_to_dict, generate_qualified_class_name


def types_match(t1, t2) -> bool:  # noqa: PLR0911
    """
    Compares two types for equivalence, handling Any, Variadic, Optional, etc.

    Returns True if t1 and t2 are considered equivalent,
    1) treating `Any` as a wildcard,
    2) unwrapping Variadic/GreedyVariadic,
    3) unwrapping Optional (i.e., Union[..., None]) to ignore the None part.
    """

    # If either side is literally `Any`, it's automatically a match
    if t1 is Any or t2 is Any:
        return True

    # Unwrap both sides until they can no longer be unwrapped
    t1 = _unwrap_all(t1)
    t2 = _unwrap_all(t2)

    # If either was unwrapped to Any, match
    if t1 is Any or t2 is Any:
        return True

    origin1 = get_origin(t1)
    origin2 = get_origin(t2)

    # If both are non-generic, compare directly
    if origin1 is None and origin2 is None:
        return t1 == t2

    # If only one side is generic, then they differ (unless we had Any, handled above)
    if (origin1 is None) != (origin2 is None):
        return False

    # If both sides have different origins (e.g., list vs. dict), mismatch
    if origin1 != origin2:
        return False

    # Compare type arguments pairwise
    args1 = get_args(t1)
    args2 = get_args(t2)
    if len(args1) != len(args2):
        return False

    return all(types_match(a1, a2) for a1, a2 in zip(args1, args2))


def _unwrap_all(t):
    """
    Unwraps a type until no more unwrapping is possible.

    Recursively unwraps:
      - Variadic[...] or GreedyVariadic[...]
      - Optional[...] (i.e., Union[..., None])
    until no more unwrapping is possible.
    """
    previous = None
    current = t

    # We'll repeatedly try unwrapping until it no longer changes
    while True:
        previous = current
        # 1) Unwrap Variadic / GreedyVariadic
        current = _unwrap_variadics(current)
        # 2) Unwrap Optional (Union[..., None])
        current = _unwrap_optionals(current)
        # If nothing changed in this pass, we're done
        if current == previous:
            return current


def _unwrap_variadics(t):
    """
    Unwrap Variadic[...] or GreedyVariadic[...] types.

    If t is Variadic[...] or GreedyVariadic[...],
    unwrap and return the inner type; otherwise return t as-is.
    """
    origin = get_origin(t)
    if origin is Variadic or origin is GreedyVariadic:
        # e.g. Variadic[List[Document]] -> List[Document]
        inner_args = get_args(t)
        if len(inner_args) == 1:
            return inner_args[0]
        # If there's more than one argument, define your own rules
        # For now, we just return the tuple of unwrapped types
        return inner_args
    return t


def _unwrap_optionals(t):
    """
    Unwrap Optional[...] types.

    If t is Optional[X] (i.e. Union[X, NoneType]),
    return just X. Otherwise return t as-is.
    """
    origin = get_origin(t)
    if origin is Union:
        args = list(get_args(t))
        # If NoneType is present, remove it
        none_type = type(None)
        if none_type in args:
            args.remove(none_type)
            # If that leaves exactly 1 type, treat it as 'X'
            if len(args) == 1:
                return args[0]
            # If it leaves more than one type, we remain a Union
            return Union[tuple(args)]
    return t


class InvalidMappingError(Exception):
    """Raised when input or output mappings are invalid or type conflicts are found."""

    pass


@component
class PipelineWrapper:
    """
    A wrapper component that encapsulates a Haystack pipeline with flexible I/O mapping.

    This wrapper allows for remapping of input and output socket names between the wrapped
    pipeline and the external interface. It handles type checking and verification of all
    mappings.

    :param pipeline: The pipeline wrapped by the component
    :param input_mapping: Mapping from wrapper input names to lists of pipeline socket paths
        in format "component_name.socket_name"
    :param output_mapping: Mapping from pipeline socket paths to wrapper output names
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
        Initialize the pipeline wrapper with optional I/O mappings.

        :param pipeline: The pipeline to wrap
        :param input_mapping: Optional input name mapping configuration
        :param output_mapping: Optional output name mapping configuration
        """
        self.pipeline: Pipeline = pipeline
        self.input_mapping: Optional[Dict[str, List[str]]] = input_mapping
        self.output_mapping: Optional[Dict[str, str]] = output_mapping
        self._set_pipeline_io()

    def _set_pipeline_io(self) -> None:
        """
        Initialize pipeline by resolving input/output types and setting them on wrapper.

        :raises InvalidMappingError: If type conflicts are found in input or output mappings
        """
        self._resolve_and_set_input_types()
        output_types = self._resolve_output_types()
        component.set_output_types(self, **output_types)

    @staticmethod
    def _split_component_path(path: str) -> Tuple[str, str]:
        """
        Split a component path into component name and socket name.

        :param path: String in format "component_name.socket_name"
        :return: Tuple of (component_name, socket_name)
        :raises InvalidMappingError: If path format is invalid
        """
        try:
            comp_name, socket_name = path.split(".")
            return comp_name, socket_name
        except ValueError:
            raise InvalidMappingError(f"Invalid path format: {path}. Expected 'component_name.socket_name'")

    def _resolve_and_set_input_types(self) -> None:
        """
        Resolve input types from pipeline and set them on wrapper.

        Examines the pipeline's declared inputs and merges them with the input mapping
        to check for type conflicts and set appropriate input types on the wrapper component.

        :raises InvalidMappingError: If type conflicts are found or mapped components/sockets
            don't exist
        """
        pipeline_inputs = self.pipeline.inputs(include_components_with_connected_inputs=False)

        if input_mapping := self.input_mapping or self.pipeline.metadata.get("input_mapping"):
            self._handle_explicit_input_mapping(pipeline_inputs, input_mapping)
        else:
            self._handle_auto_input_mapping(pipeline_inputs)

    def _handle_explicit_input_mapping(  # noqa: PLR0912
        self, pipeline_inputs: Dict[str, Dict[str, Any]], input_mapping: Dict[str, List[str]]
    ) -> None:
        """
        Handle case where explicit input mapping is provided.

        :param pipeline_inputs: Dictionary of pipeline input specifications
        :param input_mapping: Mapping from wrapper inputs to pipeline socket paths
        :raises InvalidMappingError: If mapping is invalid or type conflicts exist
        """
        for wrapper_input_name, pipeline_input_paths in input_mapping.items():
            if not isinstance(pipeline_input_paths, list):
                pipeline_input_paths = [pipeline_input_paths]

            resolved_type: Optional[Type] = None
            resolved_default: Optional[Any] = None

            for path in pipeline_input_paths:
                comp_name, socket_name = self._split_component_path(path)

                if comp_name not in pipeline_inputs:
                    raise InvalidMappingError(f"Component '{comp_name}' not found in pipeline inputs.")

                if socket_name not in pipeline_inputs[comp_name]:
                    raise InvalidMappingError(f"Input socket '{socket_name}' not found in component '{comp_name}'.")

                socket_info = pipeline_inputs[comp_name][socket_name]
                current_type = socket_info["type"]

                if resolved_type is None:
                    resolved_type = current_type
                elif not types_match(resolved_type, current_type):
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

            if resolved_type is None:
                raise InvalidMappingError(f"Could not resolve type for input '{wrapper_input_name}'")

            if resolved_default is not None:
                component.set_input_type(self, wrapper_input_name, resolved_type, default=resolved_default)
            else:
                component.set_input_type(self, wrapper_input_name, resolved_type)

    def _handle_auto_input_mapping(self, pipeline_inputs: Dict[str, Dict[str, Any]]) -> None:
        """
        Handle case where input mapping should be auto-detected.

        :param pipeline_inputs: Dictionary of pipeline input specifications
        :raises InvalidMappingError: If type conflicts exist between components
        """
        aggregated_inputs: Dict[str, Dict[str, Any]] = {}

        for inputs_dict in pipeline_inputs.values():
            for socket_name, socket_info in inputs_dict.items():
                if socket_name not in aggregated_inputs:
                    aggregated_inputs[socket_name] = socket_info
                else:
                    existing_type = aggregated_inputs[socket_name]["type"]
                    if not types_match(existing_type, socket_info["type"]):
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
                    self, input_name, info["type"], default=default_val if default_val is not None else None
                )

    def _resolve_output_types(self) -> Dict[str, Any]:
        """
        Resolve output types based on pipeline outputs and mapping.

        :return: Dictionary mapping wrapper output names to their types
        :raises InvalidMappingError: If mapping is invalid or output conflicts exist
        """
        pipeline_outputs = self.pipeline.outputs(include_components_with_connected_outputs=False)

        if output_mapping := self.output_mapping or self.pipeline.metadata.get("output_mapping"):
            return self._handle_explicit_output_mapping(pipeline_outputs, output_mapping)
        else:
            return self._handle_auto_output_mapping(pipeline_outputs)

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
            comp_name, socket_name = self._split_component_path(pipeline_output_path)

            if comp_name not in pipeline_outputs:
                raise InvalidMappingError(f"Component '{comp_name}' not found among pipeline outputs.")
            if socket_name not in pipeline_outputs[comp_name]:
                raise InvalidMappingError(f"Output socket '{socket_name}' not found in component '{comp_name}'.")
            if wrapper_output_name in resolved_outputs:
                raise InvalidMappingError(f"Duplicate wrapper output name '{wrapper_output_name}' in output_mapping.")

            resolved_outputs[wrapper_output_name] = pipeline_outputs[comp_name][socket_name]["type"]

        return resolved_outputs

    @staticmethod
    def _handle_auto_output_mapping(pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Handle case where output mapping should be auto-detected.

        :param pipeline_outputs: Dictionary of pipeline output specifications
        :return: Dictionary of resolved output types
        :raises InvalidMappingError: If output conflicts exist
        """
        resolved_outputs = {}
        used_output_names: set[str] = set()

        for outputs_dict in pipeline_outputs.values():
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
        if not self.pipeline:
            raise ValueError("No pipeline configured. Provide a valid pipeline before calling run().")

        pipeline_inputs = self._prepare_pipeline_inputs(kwargs)
        pipeline_outputs = self.pipeline.run(data=pipeline_inputs)
        return self._process_pipeline_outputs(pipeline_outputs)

    def _prepare_pipeline_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Prepare inputs for pipeline execution.

        :param inputs: Input arguments provided to wrapper
        :return: Dictionary of pipeline inputs in required format
        """
        if input_mapping := self.input_mapping or self.pipeline.metadata.get("input_mapping"):
            return self._map_explicit_inputs(input_mapping=input_mapping, inputs=inputs)
        else:
            return self._map_auto_inputs(inputs)

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

            if not isinstance(pipeline_input_paths, list):
                pipeline_input_paths = [pipeline_input_paths]

            for socket_path in pipeline_input_paths:
                comp_name, input_name = self._split_component_path(socket_path)
                if comp_name not in pipeline_inputs:
                    pipeline_inputs[comp_name] = {}
                pipeline_inputs[comp_name][input_name] = inputs[wrapper_input_name]

        return pipeline_inputs

    def _map_auto_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Map inputs automatically based on matching names.

        :param inputs: Input arguments provided to wrapper
        :return: Dictionary of mapped pipeline inputs
        """
        pipeline_inputs: Dict[str, Dict[str, Any]] = {}
        pipeline_decl_inputs = self.pipeline.inputs(include_components_with_connected_inputs=False)

        for comp_name, inputs_dict in pipeline_decl_inputs.items():
            for socket_name in inputs_dict.keys():
                if socket_name in inputs:
                    pipeline_inputs.setdefault(comp_name, {})[socket_name] = inputs[socket_name]

        return pipeline_inputs

    def _process_pipeline_outputs(self, pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process outputs from pipeline execution.

        :param pipeline_outputs: Raw outputs from pipeline execution
        :return: Dictionary of processed outputs
        :raises InvalidMappingError: If output conflicts occur during auto-mapping
        """
        if output_mapping := self.output_mapping or self.pipeline.metadata.get("output_mapping"):
            return self._map_explicit_outputs(pipeline_outputs, output_mapping)
        else:
            return self._map_auto_outputs(pipeline_outputs)

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

    @staticmethod
    def _map_auto_outputs(pipeline_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Map outputs automatically based on matching names.

        :param pipeline_outputs: Raw outputs from pipeline execution
        :return: Dictionary of mapped outputs
        :raises InvalidMappingError: If output conflicts occur during auto-mapping
        """
        outputs: Dict[str, Any] = {}
        seen: set[str] = set()

        for comp_output_dict in pipeline_outputs.values():
            for socket_name, value in comp_output_dict.items():
                if socket_name in seen:
                    raise InvalidMappingError(
                        f"Output conflict: output '{socket_name}' is produced by multiple components. "
                        "Please provide an output_mapping to disambiguate."
                    )
                outputs[socket_name] = value
                seen.add(socket_name)

        return outputs

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pipeline wrapper to a dictionary representation.

        :return: Dictionary containing serialized pipeline wrapper data
        """
        serialized_pipeline = self.pipeline.to_dict()
        return default_to_dict(
            self, pipeline=serialized_pipeline, input_mapping=self.input_mapping, output_mapping=self.output_mapping
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineWrapper":
        """
        Create a pipeline wrapper instance from a dictionary representation.

        :param data: Dictionary containing serialized pipeline wrapper data
        :return: New PipelineWrapper instance
        """
        pipeline = Pipeline.from_dict(data["init_parameters"]["pipeline"])
        data["init_parameters"]["pipeline"] = pipeline

        return default_from_dict(cls, data)

    def to_pipeline_wrapper_dict(self) -> Dict[str, Any]:
        """
        Convert to a pipeline wrapper dictionary representation.

        :return: Dictionary containing serialized pipeline wrapper data
        """
        serialized_pipeline = self.pipeline.to_dict()
        serialized = default_to_dict(
            self, pipeline=serialized_pipeline, input_mapping=self.input_mapping, output_mapping=self.output_mapping
        )
        serialized["type"] = generate_qualified_class_name(PipelineWrapper)
        return serialized
