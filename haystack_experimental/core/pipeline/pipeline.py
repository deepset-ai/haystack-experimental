# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=too-many-return-statements, too-many-positional-arguments


from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

from haystack import logging, tracing
from haystack.core.pipeline.base import ComponentPriority
from haystack.core.pipeline.pipeline import Pipeline as HaystackPipeline
from haystack.telemetry import pipeline_running
from haystack.utils import _deserialize_value_with_schema

from haystack_experimental.core.errors import PipelineInvalidResumeStateError
from haystack_experimental.core.pipeline.base import PipelineBase

from ...components.agents import Agent
from ...dataclasses.breakpoints import AgentBreakpoint, Breakpoint
from .breakpoint import (
    _validate_break_point,
    _validate_components_against_pipeline,
    check_regular_break_point,
    handle_agent_break_point,
    trigger_break_point,
)

logger = logging.getLogger(__name__)


# We inherit from both HaystackPipeline and PipelineBase to ensure that we have the
# necessary methods and properties from both classes.
class Pipeline(HaystackPipeline, PipelineBase):
    """
    Synchronous version of the orchestration engine.

    Orchestrates component execution according to the execution graph, one after the other.
    """

    def _handle_resume_state(self, resume_state: Dict[str, Any]) -> tuple[Dict[str, int], Dict[str, Any], bool, list]:
        """
        Handle resume state initialization.

        :param resume_state: The resume state to handle
        :return: Tuple of (component_visits, data, resume_agent_in_pipeline, ordered_component_names)
        """
        if resume_state.get("agent_name"):
            return self._handle_agent_resume_state(resume_state)
        else:
            return self._handle_regular_resume_state(resume_state)

    def _handle_agent_resume_state(
        self, resume_state: Dict[str, Any]
    ) -> tuple[Dict[str, int], Dict[str, Any], bool, list]:
        """
        Handle agent-specific resume state.

        :param resume_state: The resume state to handle
        :return: Tuple of (component_visits, data, resume_agent_in_pipeline, ordered_component_names)
        """
        agent_name = resume_state["agent_name"]
        for name, component in self.graph.nodes.items():
            if component["instance"].__class__.__name__ == "Agent" and name == agent_name:
                main_pipeline_state = resume_state.get("main_pipeline_state", {})
                component_visits = main_pipeline_state.get("component_visits", {})
                ordered_component_names = main_pipeline_state.get("ordered_component_names", [])
                data = _deserialize_value_with_schema(main_pipeline_state.get("inputs", {}))
                return component_visits, data, True, ordered_component_names

        # Fallback to regular resume if agent not found
        return self._handle_regular_resume_state(resume_state)

    def _handle_regular_resume_state(
        self, resume_state: Dict[str, Any]
    ) -> tuple[Dict[str, int], Dict[str, Any], bool, list]:
        """
        Handle regular component resume state.

        :param resume_state: The resume state to handle
        :return: Tuple of (component_visits, data, resume_agent_in_pipeline, ordered_component_names)
        """
        component_visits, data, resume_state, ordered_component_names = self.inject_resume_state_into_graph(
            resume_state=resume_state,
        )
        data = _deserialize_value_with_schema(resume_state["pipeline_state"]["inputs"])
        return component_visits, data, False, ordered_component_names

    def run(  # noqa: PLR0915, PLR0912
        self,
        data: Dict[str, Any],
        include_outputs_from: Optional[Set[str]] = None,
        break_point: Optional[Union[Breakpoint, AgentBreakpoint]] = None,
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

        :param break_point:
            A set of breakpoints that can be used to debug the pipeline execution.

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
            When a pipeline_breakpoint is triggered. Contains the component name, state, and partial results.
        """
        pipeline_running(self)

        if break_point and resume_state:
            msg = (
                "pipeline_breakpoint and resume_state cannot be provided at the same time. "
                "The pipeline run will be aborted."
            )
            raise PipelineInvalidResumeStateError(message=msg)

        # make sure all breakpoints are valid, i.e. reference components in the pipeline
        if break_point:
            _validate_break_point(break_point, self.graph)

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        if include_outputs_from is None:
            include_outputs_from = set()

        if not resume_state:
            # normalize `data`
            data = self._prepare_component_input_data(data)

            # Raise ValueError if input is malformed in some way
            self.validate_input(data)

            # We create a list of components in the pipeline sorted by name, so that the algorithm runs
            # deterministically and independent of insertion order into the pipeline.
            ordered_component_names = sorted(self.graph.nodes.keys())

            # We track component visits to decide if a component can run.
            component_visits = dict.fromkeys(ordered_component_names, 0)
            resume_agent_in_pipeline = False

        else:
            # Handle resume state
            component_visits, data, resume_agent_in_pipeline, ordered_component_names = self._handle_resume_state(
                resume_state
            )

        cached_topological_sort = None
        # We need to access a component's receivers multiple times during a pipeline run.
        # We store them here for easy access.
        cached_receivers = {name: self._find_receivers_from(name) for name in ordered_component_names}

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
            priority_queue = self._fill_queue(ordered_component_names, inputs, component_visits)

            # check if pipeline is blocked before execution
            self.validate_pipeline(priority_queue)

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

                is_resume = bool(resume_state and resume_state["pipeline_breakpoint"]["component"] == component_name)
                component_inputs = self._consume_component_inputs(
                    component_name=component_name, component=component, inputs=inputs, is_resume=is_resume
                )

                # We need to add missing defaults using default values from input sockets because the run signature
                # might not provide these defaults for components with inputs defined dynamically upon component
                # initialization
                component_inputs = self._add_missing_input_defaults(component_inputs, component["input_sockets"])

                # Scenario 1: Resume state is provided to resume the pipeline at a specific component
                # Deserialize the component_inputs if they are passed in resume state
                # this check will prevent other component_inputs generated at runtime from being deserialized
                if resume_state and component_name in resume_state["pipeline_state"]["inputs"].keys():
                    for key, value in component_inputs.items():
                        component_inputs[key] = _deserialize_value_with_schema(value)

                # Scenario 2: a breakpoint is provided to stop the pipeline at a specific component and visit count
                breakpoint_triggered = False
                if break_point is not None:
                    agent_breakpoint = False

                    if isinstance(break_point, AgentBreakpoint):
                        component_instance = component["instance"]
                        if isinstance(component_instance, Agent):
                            component_inputs = handle_agent_break_point(
                                break_point,
                                component_name,
                                component_inputs,
                                inputs,
                                component_visits,
                                ordered_component_names,
                                data,
                                debug_path,
                            )
                            agent_breakpoint = True

                    if not agent_breakpoint and isinstance(break_point, Breakpoint):
                        breakpoint_triggered = check_regular_break_point(break_point, component_name, component_visits)

                    if breakpoint_triggered:
                        trigger_break_point(
                            component_name,
                            component_inputs,
                            inputs,
                            component_visits,
                            debug_path,
                            data,
                            ordered_component_names,
                            pipeline_outputs,
                        )

                if resume_agent_in_pipeline:
                    # inject the resume_state into the component (the Agent) inputs
                    component_inputs["resume_state"] = resume_state
                    component_inputs["break_point"] = None

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
                    priority_queue = self._fill_queue(ordered_component_names, inputs, component_visits)

            if break_point and not agent_breakpoint:
                logger.warning(
                    "The given breakpoint {break_point} was never triggered. This is because:\n"
                    "1. The provided component is not a part of the pipeline execution path.\n"
                    "2. The component did not reach the visit count specified in the pipeline_breakpoint",
                    pipeline_breakpoint=break_point,
                )

            return pipeline_outputs

    def inject_resume_state_into_graph(self, resume_state):
        """
        Loads the resume state from a file and injects it into the pipeline graph.

        """
        # We previously check if the resume_state is None but this is needed to prevent a typing error
        if not resume_state:
            raise PipelineInvalidResumeStateError("Cannot inject resume state: resume_state is None")

        # check if the resume_state is valid for the current pipeline
        _validate_components_against_pipeline(resume_state, self.graph)

        data = self._prepare_component_input_data(resume_state["pipeline_state"]["inputs"])
        component_visits = resume_state["pipeline_state"]["component_visits"]
        ordered_component_names = resume_state["pipeline_state"]["ordered_component_names"]
        logger.info(
            "Resuming pipeline from {component} with visit count {visits}",
            component=resume_state["pipeline_breakpoint"]["component"],
            visits=resume_state["pipeline_breakpoint"]["visits"],
        )

        return component_visits, data, resume_state, ordered_component_names
