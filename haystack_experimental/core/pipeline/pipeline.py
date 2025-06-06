# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union, cast

from haystack import Answer, Document, ExtractedAnswer, logging, tracing
from haystack.core.component import Component
from haystack.core.pipeline.component_checks import is_socket_lazy_variadic
from haystack.dataclasses import ChatMessage, GeneratedAnswer, SparseEmbedding
from haystack.telemetry import pipeline_running

from haystack_experimental.core.errors import (
    PipelineBreakpointException,
    PipelineInvalidResumeStateError,
    PipelineRuntimeError,
)
from haystack_experimental.core.pipeline.base import (
    _COMPONENT_INPUT,
    _COMPONENT_OUTPUT,
    _COMPONENT_VISITS,
    ComponentPriority,
    PipelineBase,
)

logger = logging.getLogger(__name__)


class Pipeline(PipelineBase):
    """
    Synchronous version of the orchestration engine.

    Orchestrates component execution according to the execution graph, one after the other.
    """

    @staticmethod
    def _run_component(
        component_name: str,
        component: Dict[str, Any],
        inputs: Dict[str, Any],
        component_visits: Dict[str, int],
        parent_span: Optional[tracing.Span] = None,
    ) -> Dict[str, Any]:
        """
        Runs a Component with the given inputs.

        :param component_name: Name of the Component.
        :param component: Component with component metadata.
        :param inputs: Inputs for the Component.
        :param component_visits: Current state of component visits.
        :param parent_span: The parent span to use for the newly created span.
                            This is to allow tracing to be correctly linked to the pipeline run.

        :raises:
            PipelineRuntimeError: If Component doesn't return a dictionary.

        :returns:
            The output of the Component.
        """
        instance: Component = component["instance"]

        with PipelineBase._create_component_span(
            component_name=component_name, instance=instance, inputs=inputs, parent_span=parent_span
        ) as span:
            # We deepcopy the inputs otherwise we might lose that information
            # when we delete them in case they're sent to other Components
            span.set_content_tag(_COMPONENT_INPUT, deepcopy(inputs))
            logger.info("Running component {component_name}", component_name=component_name)

            try:
                component_output = instance.run(**inputs)
            except Exception as error:
                raise PipelineRuntimeError.from_exception(component_name, instance.__class__, error) from error
            component_visits[component_name] += 1

            if not isinstance(component_output, Mapping):
                raise PipelineRuntimeError.from_invalid_output(component_name, instance.__class__, component_output)

            span.set_tag(_COMPONENT_VISITS, component_visits[component_name])
            span.set_content_tag(_COMPONENT_OUTPUT, component_output)

            return cast(Dict[Any, Any], component_output)

    def run(  # noqa: PLR0915, PLR0912
        self,
        data: Dict[str, Any],
        include_outputs_from: Optional[Set[str]] = None,
        breakpoints: Optional[Set[Tuple[str, Optional[int]]]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
        debug_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the Pipeline with given input data.

        Usage:
        ```python
        from haystack import Pipeline, Document
        from haystack.utils import Secret
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
        from haystack.components.generators import OpenAIGenerator
        from haystack.components.builders.answer_builder import AnswerBuilder
        from haystack.components.builders.prompt_builder import PromptBuilder

        # Write documents to InMemoryDocumentStore
        document_store = InMemoryDocumentStore()
        document_store.write_documents([
            Document(content="My name is Jean and I live in Paris."),
            Document(content="My name is Mark and I live in Berlin."),
            Document(content="My name is Giorgio and I live in Rome.")
        ])

        prompt_template = \"\"\"
        Given these documents, answer the question.
        Documents:
        {% for doc in documents %}
            {{ doc.content }}
        {% endfor %}
        Question: {{question}}
        Answer:
        \"\"\"

        retriever = InMemoryBM25Retriever(document_store=document_store)
        prompt_builder = PromptBuilder(template=prompt_template)
        llm = OpenAIGenerator(api_key=Secret.from_token(api_key))

        rag_pipeline = Pipeline()
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("llm", llm)
        rag_pipeline.connect("retriever", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder", "llm")

        # Ask a question
        question = "Who lives in Paris?"
        results = rag_pipeline.run(
            {
                "retriever": {"query": question},
                "prompt_builder": {"question": question},
            }
        )

        print(results["llm"]["replies"])
        # Jean lives in Paris
        ```

        :param data:
            A dictionary of inputs for the pipeline's components. Each key is a component name
            and its value is a dictionary of that component's input parameters:
            ```
            data = {
                "comp1": {"input1": 1, "input2": 2},
            }
            ```
            For convenience, this format is also supported when input names are unique:
            ```
            data = {
                "input1": 1, "input2": 2,
            }
            ```
        :param include_outputs_from:
            Set of component names whose individual outputs are to be
            included in the pipeline's output. For components that are
            invoked multiple times (in a loop), only the last-produced
            output is included.

        :param breakpoints:
            Set of tuples of component names and visit counts at which the pipeline should break execution.
            If the visit count is not given, it is assumed to be 0, it will break on the first visit.

        :param resume_state:
            A dictionary containing the state of a previously saved pipeline execution.

        :param debug_path:
            Path to the directory where the pipeline state should be saved.

        :returns:
            A dictionary where each entry corresponds to a component name
            and its output. If `include_outputs_from` is `None`, this dictionary
            will only contain the outputs of leaf components, i.e., components
            without outgoing connections.

        :raises ValueError:
            If invalid inputs are provided to the pipeline.
        :raises PipelineRuntimeError:
            If the Pipeline contains cycles with unsupported connections that would cause
            it to get stuck and fail running.
            Or if a Component fails or returns output in an unsupported type.
        :raises PipelineMaxComponentRuns:
            If a Component reaches the maximum number of times it can be run in this Pipeline.
        :raises PipelineBreakpointException:
            When a breakpoint is triggered. Contains the component name, state, and partial results.
        """
        pipeline_running(self)

        if breakpoints and resume_state:
            logger.warning(
                "Breakpoints cannot be provided when resuming a pipeline. All breakpoints will be ignored.",
            )
        self.debug_path = debug_path
        self.resume_state = resume_state

        # make sure breakpoints are valid and have a default visit count
        validated_breakpoints = self._validate_breakpoints(breakpoints) if breakpoints else None

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        if include_outputs_from is None:
            include_outputs_from = set()

        if not self.resume_state:
            # normalize `data`
            data = self._prepare_component_input_data(data)

            # Raise ValueError if input is malformed in some way
            self._validate_input(data)

            # We create a list of components in the pipeline sorted by name, so that the algorithm runs
            # deterministically and independent of insertion order into the pipeline.
            self.ordered_component_names = sorted(self.graph.nodes.keys())

            # We track component visits to decide if a component can run.
            component_visits = dict.fromkeys(self.ordered_component_names, 0)

        else:
            # inject the resume state into the graph
            component_visits, data = self.inject_resume_state_into_graph()

        cached_topological_sort = None
        # We need to access a component's receivers multiple times during a pipeline run.
        # We store them here for easy access.
        cached_receivers = {name: self._find_receivers_from(name) for name in self.ordered_component_names}

        pipeline_outputs: Dict[str, Any] = {}
        with tracing.tracer.trace(
            "haystack.pipeline.run",
            tags={
                "haystack.pipeline.input_data": data,
                "haystack.pipeline.output_data": pipeline_outputs,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_runs_per_component": self._max_runs_per_component,
            },
        ) as span:
            inputs = self._convert_to_internal_format(pipeline_inputs=data)
            priority_queue = self._fill_queue(self.ordered_component_names, inputs, component_visits)

            # check if pipeline is blocked before execution
            self.validate_pipeline(priority_queue)

            try:
                while True:
                    candidate = self._get_next_runnable_component(priority_queue, component_visits)
                    if candidate is None:
                        break

                    priority, component_name, component = candidate

                    if len(priority_queue) > 0 and priority in [ComponentPriority.DEFER, ComponentPriority.DEFER_LAST]:
                        component_name, topological_sort = self._tiebreak_waiting_components(
                            component_name=component_name,
                            priority=priority,
                            priority_queue=priority_queue,
                            topological_sort=cached_topological_sort,
                        )
                        cached_topological_sort = topological_sort
                        component = self._get_component_with_graph_metadata_and_visits(
                            component_name, component_visits[component_name]
                        )

                    # this breaks the pipeline breakpoints
                    component_inputs = self._consume_component_inputs(
                        component_name=component_name, component=component, inputs=inputs
                    )
                    # We need to add missing defaults using default values from input sockets because the run signature
                    # might not provide these defaults for components with inputs defined dynamically upon component
                    # initialization
                    component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

                    # Deserialize the component_inputs if they are passed in resume state
                    # this check will prevent other component_inputs generated at runtime from being deserialized
                    if self.resume_state and component_name in self.resume_state["pipeline_state"]["inputs"].keys():
                        for key, value in component_inputs.items():
                            component_inputs[key] = _deserialize_component_input(value)

                    if validated_breakpoints and not self.resume_state:
                        state_inputs_serialised = deepcopy(inputs)
                        # inject the component_inputs into the state_inputs so we can this component init params in
                        # the JSON state
                        state_inputs_serialised[component_name] = deepcopy(component_inputs)

                        # we use dict instead of to_dict() because it strips away class types
                        # We use copy instead of deepcopy to avoid issues with unpickleable objects like RLock
                        params = copy(component["instance"].__dict__)
                        state_inputs_serialised[component_name]["init_parameters"] = params

                        if "_template_string" in params:
                            params["template"] = params["_template_string"]
                            params.pop("_template_string")

                        print("PARAMS")
                        print(params)

                        Pipeline._check_breakpoints(
                            breakpoints=validated_breakpoints,
                            component_name=component_name,
                            component_visits=component_visits,
                            inputs=state_inputs_serialised,
                            debug_path=self.debug_path,
                            original_input_data=data,
                            ordered_component_names=self.ordered_component_names,
                        )
                    # the _consume_component_inputs() when applied to the DocumentJoiner inputs wraps 'documents' in an
                    # extra list, so there's a 3 level deep list, we need to flatten it to 2 levels only

                    if self.resume_state and self.resume_state["breakpoint"]["component"] == component_name:
                        for socket_name, socket in component["input_sockets"].items():
                            if is_socket_lazy_variadic(socket):
                                component_inputs[socket_name] = component_inputs[socket_name][0]

                    component_outputs = self._run_component(
                        component_name=component_name,
                        component=component,
                        inputs=component_inputs,  # the inputs to the current component
                        component_visits=component_visits,
                        parent_span=span,
                    )

                    # Updates global input state with component outputs and returns outputs that should go to
                    # pipeline outputs.
                    component_pipeline_outputs = self._write_component_outputs(
                        component_name=component_name,
                        component_outputs=component_outputs,
                        inputs=inputs,
                        receivers=cached_receivers[component_name],
                        include_outputs_from=include_outputs_from,
                    )

                    if component_pipeline_outputs:
                        pipeline_outputs[component_name] = deepcopy(component_pipeline_outputs)
                    if self._is_queue_stale(priority_queue):
                        priority_queue = self._fill_queue(self.ordered_component_names, inputs, component_visits)

            except PipelineBreakpointException as e:
                # Add the current pipeline results to the exception
                e.results = pipeline_outputs
                raise

            if breakpoints:
                logger.warning(f"Given breakpoint {breakpoints} was never triggered. This is because:")
                logger.warning("1. The provided component is not a part of the pipeline execution path.")
                logger.warning("2. The component did not reach the visit count specified in the breakpoint")
            return pipeline_outputs

    def inject_resume_state_into_graph(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, Any]]:
        """
        Loads the resume state from a file and injects it into the pipeline graph.

        """
        # We previously check if the resume_state is None but
        # this is needed to prevent a typing error
        if not self.resume_state:
            raise PipelineInvalidResumeStateError("Cannot inject resume state: resume_state is None")

        self._validate_pipeline_state(self.resume_state)
        data = self._prepare_component_input_data(self.resume_state["pipeline_state"]["inputs"])
        component_visits = self.resume_state["pipeline_state"]["component_visits"]
        self.ordered_component_names = self.resume_state["pipeline_state"]["ordered_component_names"]
        msg = (
            f"Resuming pipeline from {self.resume_state['breakpoint']['component']} "
            f"visit count {self.resume_state['breakpoint']['visits']}"
        )
        logger.info(msg)
        return component_visits, data

    def _validate_breakpoints(self, breakpoints: Set[Tuple[str, Optional[int]]]) -> Set[Tuple[str, int]]:
        """
        Validates the breakpoints passed to the pipeline.

        Make sure they are all valid components registered in the pipeline,
        If the visit is not given, it is assumed to be 0, it will break on the first visit.

        :param breakpoints: Set of tuples of component names and visit counts at which the pipeline should stop.
        :returns:
            Set of valid breakpoints.
        """

        processed_breakpoints: Set[Tuple[str, int]] = set()

        for break_point in breakpoints:
            if break_point[0] not in self.graph.nodes:
                raise ValueError(f"Breakpoint {break_point} is not a registered component in the pipeline")
            valid_breakpoint: Tuple[str, int] = (break_point[0], 0 if break_point[1] is None else break_point[1])
            processed_breakpoints.add(valid_breakpoint)
        return processed_breakpoints

    def _validate_pipeline_state(self, resume_state: Dict[str, Any]) -> None:
        """
        Validates that the resume_state contains valid configuration for the current pipeline.

        Raises a PipelineRuntimeError if any component is missing or if the state structure is invalid.

        :param resume_state: The saved state to validate.
        """

        pipeline_state = resume_state["pipeline_state"]
        valid_components = set(self.graph.nodes.keys())

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
                f"Invalid resume state: components {missing_input} in 'input_data' "
                f"are not part of the current pipeline."
            )

        # Validate 'component_visits'
        missing_visits = set(pipeline_state["component_visits"].keys()) - valid_components
        if missing_visits:
            raise PipelineInvalidResumeStateError(
                f"Invalid resume state: components {missing_visits} in 'component_visits' "
                f"are not part of the current pipeline."
            )

        logger.info(
            f"Resuming pipeline from component: {resume_state['breakpoint']['component']} "
            f"(visit {resume_state['breakpoint']['visits']})"
        )

    @staticmethod
    def _validate_resume_state(state: Dict[str, Any]) -> None:
        """
        Validates the loaded pipeline state.

        Ensures that the state contains required keys: "input_data", "breakpoint", and "pipeline_state".

        Raises:
            ValueError: If required keys are missing or the component sets are inconsistent.
        """

        # top-level state has all required keys
        required_top_keys = {"input_data", "breakpoint", "pipeline_state"}
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

    @staticmethod
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
            Pipeline._validate_resume_state(state=state)
        except ValueError as e:
            raise ValueError(f"Invalid pipeline state from {file_path}: {str(e)}")

        logger.info(f"Successfully loaded pipeline state from: {file_path}")
        return state

    @staticmethod
    def save_state(
        inputs: Dict[str, Any],
        component_name: str,
        component_visits: Dict[str, int],
        callback_fun: Optional[Callable[..., Any]] = None,
        debug_path: Optional[Union[str, Path]] = None,
        original_input_data: Optional[Dict[str, Any]] = None,
        ordered_component_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        If a debug_path is given it saves the JSON state of the pipeline at a given component visit count in a file.

        If debug_path is not given, it returns the JSON state as a dictionary without saving it to a file.

        :raises:
            Exception: If the debug_path is not a string or a Path object, or if saving the JSON state fails.

        :returns:
            The saved state dictionary
        """
        dt = datetime.now()

        # we store a input params passed during init() in the saved state
        # this is helpful for retaining the state of the component and manual debugging
        for value in inputs.values():
            if "init_parameters" not in value:
                continue
            init_params = value.pop("init_parameters")
            for k, v in value.items():
                if k in init_params and not v:
                    value[k] = _serialize_component_input(init_params[k])

        state = {
            "input_data": _serialize_component_input(original_input_data),  # original input data
            "timestamp": dt.isoformat(),
            "breakpoint": {"component": component_name, "visits": component_visits[component_name]},
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

    @staticmethod
    def _check_breakpoints(
        breakpoints: Set[Tuple[str, int]],
        component_name: str,
        component_visits: Dict[str, int],
        inputs: Dict[str, Any],
        debug_path: Optional[Union[str, Path]] = None,
        original_input_data: Optional[Dict[str, Any]] = None,
        ordered_component_names: Optional[List[str]] = None,
    ) -> None:
        """
        Check if the `component_name` is in the breakpoints and if it should break.

        :param breakpoints: Set of tuples of component names and visit counts at which the pipeline should stop.
        :param component_name: Name of the component to check.
        :param component_visits: The number of times the component has been visited.
        :param inputs: The inputs to the pipeline.
        :param debug_path: The file path where debug state is saved.
        :param original_input_data: The original input data to the pipeline.
        :param ordered_component_names: The ordered component names in the pipeline.

        :raises PipelineBreakpointException: When a breakpoint is triggered, with component state information.
        """
        matching_breakpoints = [bp for bp in breakpoints if bp[0] == component_name]

        for bp in matching_breakpoints:
            visit_count = bp[1]
            # break only if the visit count is the same
            if visit_count == component_visits[component_name]:
                msg = f"Breaking at component {component_name} visit count {component_visits[component_name]}"
                logger.info(msg)
                state = Pipeline.save_state(
                    inputs=inputs,
                    component_name=str(component_name),
                    component_visits=component_visits,
                    debug_path=debug_path,
                    original_input_data=original_input_data,
                    ordered_component_names=ordered_component_names,
                )
                raise PipelineBreakpointException(msg, component=component_name, state=state)

    @staticmethod
    def _check_breakpoints_new(
        breakpoints: Set[Tuple[str, int]],
        component_name: str,
        component_visits: Dict[str, int],
        inputs: Dict[str, Any],
        debug_path: Optional[Union[str, Path]] = None,
        original_input_data: Optional[Dict[str, Any]] = None,
        ordered_component_names: Optional[List[str]] = None,
    ) -> None:
        """
        Check if the `component_name` is in the breakpoints and if it should break.

        :param breakpoints: Set of tuples of component names and visit counts at which the pipeline should stop.
        :param component_name: Name of the component to check.
        :param component_visits: The number of times the component has been visited.
        :param inputs: The inputs to the pipeline.
        :param debug_path: The file path where debug state is saved.
        :param original_input_data: The original input data to the pipeline.
        :param ordered_component_names: The ordered component names in the pipeline.

        :raises PipelineBreakpointException: When a breakpoint is triggered, with component state information.
        """
        for component, visit_count in breakpoints:
            if component != component_name:
                continue

            if visit_count == component_visits[component_name]:
                msg = f"Breaking at component {component_name} visit count {component_visits[component_name]}"
                logger.info(msg)
                state = Pipeline.save_state(
                    inputs=inputs,
                    component_name=str(component_name),
                    component_visits=component_visits,
                    debug_path=debug_path,
                    original_input_data=original_input_data,
                    ordered_component_names=ordered_component_names,
                )
                raise PipelineBreakpointException(msg, component=component_name, state=state)


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
