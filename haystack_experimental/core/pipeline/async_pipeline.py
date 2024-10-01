# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Set, Tuple, Union
from warnings import warn

from haystack import logging, tracing
from haystack.core.component import Component
from haystack.core.errors import PipelineMaxComponentRuns, PipelineRuntimeError
from haystack.core.pipeline.base import (
    PipelineBase,
    _add_missing_input_defaults,
    _dequeue_component,
    _dequeue_waiting_component,
    _enqueue_component,
    _enqueue_waiting_component,
    _is_lazy_variadic,
)
from haystack.telemetry import pipeline_running

logger = logging.getLogger(__name__)


class AsyncPipeline(PipelineBase):
    """
    Asynchronous version of the orchestration engine.

    The primary difference between this and the synchronous version is that this version
    will attempt to execute a component's async `run_async` method if it exists. If it doesn't,
    the synchronous `run` method is executed as an awaitable task on a thread pool executor. This
    version also eagerly yields the output of each component as soon as it is available.
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        max_runs_per_component: int = 100,
        async_executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Creates the asynchronous Pipeline.

        :param metadata:
            Arbitrary dictionary to store metadata about this `Pipeline`. Make sure all the values contained in
            this dictionary can be serialized and deserialized if you wish to save this `Pipeline` to file.
        :param max_runs_per_component:
            How many times the `Pipeline` can run the same Component.
            If this limit is reached a `PipelineMaxComponentRuns` exception is raised.
            If not set defaults to 100 runs per Component.
        :param async_executor:
            Optional ThreadPoolExecutor to use for running synchronous components. If not provided, a single-threaded
            executor will initialized and used.
        """
        super().__init__(metadata, max_runs_per_component)

        # We only need one thread as we'll immediately block after launching it.
        self.executor = (
            ThreadPoolExecutor(
                thread_name_prefix=f"async-pipeline-executor-{id(self)}", max_workers=1
            )
            if async_executor is None
            else async_executor
        )

    async def _run_component(self, name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs a Component with the given inputs.

        :param name: Name of the Component as defined in the Pipeline.
        :param inputs: Inputs for the Component.
        :raises PipelineRuntimeError: If Component doesn't return a dictionary.
        :return: The output of the Component.
        """
        instance: Component = self.graph.nodes[name]["instance"]

        with tracing.tracer.trace(
            "haystack.component.run",
            tags={
                "haystack.component.name": name,
                "haystack.component.type": instance.__class__.__name__,
                "haystack.component.input_types": {
                    k: type(v).__name__ for k, v in inputs.items()
                },
                "haystack.component.input_spec": {
                    key: {
                        "type": (
                            value.type.__name__
                            if isinstance(value.type, type)
                            else str(value.type)
                        ),
                        "senders": value.senders,
                    }
                    for key, value in instance.__haystack_input__._sockets_dict.items()  # type: ignore
                },
                "haystack.component.output_spec": {
                    key: {
                        "type": (
                            value.type.__name__
                            if isinstance(value.type, type)
                            else str(value.type)
                        ),
                        "receivers": value.receivers,
                    }
                    for key, value in instance.__haystack_output__._sockets_dict.items()  # type: ignore
                },
            },
        ) as span:
            span.set_content_tag("haystack.component.input", inputs)

            res: Dict[str, Any]
            if instance.__haystack_supports_async__:  # type: ignore
                logger.info(
                    "Running async component {component_name}", component_name=name
                )
                res = await instance.run_async(**inputs)  # type: ignore
            else:
                logger.info(
                    "Running sync component {component_name} on executor",
                    component_name=name,
                )
                res = await asyncio.get_event_loop().run_in_executor(
                    self.executor, lambda: instance.run(**inputs)
                )

            self.graph.nodes[name]["visits"] += 1

            # After a Component that has variadic inputs is run, we need to reset the variadic inputs that were consumed
            for socket in instance.__haystack_input__._sockets_dict.values():  # type: ignore
                if socket.name not in inputs:
                    continue
                if socket.is_variadic:
                    inputs[socket.name] = []

            if not isinstance(res, Mapping):
                raise PipelineRuntimeError(
                    f"Component '{name}' didn't return a dictionary. "
                    "Components must always return dictionaries: check the documentation."
                )
            span.set_tag("haystack.component.visits", self.graph.nodes[name]["visits"])
            span.set_content_tag("haystack.component.output", res)

            return res

    async def run(  # noqa: PLR0915
        self,
        data: Dict[str, Any],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Runs the pipeline with given input data.

        Since the return value of this function is an asynchroneous generator, the
        execution will only progress if the generator is consumed.


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
                "input1": 1,
                "input2": 2,
            }
            ```

        :yields:
            A dictionary where each entry corresponds to a component name and its
            output. Outputs of each component are yielded as soon as they are available,
            and the final output is yielded as the final dictionary.

        :raises PipelineRuntimeError:
            If a component fails or returns unexpected output.

        Example a - Using named components:
        Consider a 'Hello' component that takes a 'word' input and outputs a greeting.

        ```python
        @component
        class Hello:
            @component.output_types(output=str)
            def run(self, word: str):
                return {"output": f"Hello, {word}!"}
        ```

        Create a pipeline with two 'Hello' components connected together:

        ```python
        pipeline = Pipeline()
        pipeline.add_component("hello", Hello())
        pipeline.add_component("hello2", Hello())
        pipeline.connect("hello.output", "hello2.word")

        async for result in pipeline.run(data={"hello": {"word": "world"}}):
            print(result)
        ```

        This will return the results in the following order:
            {"hello": "Hello, world!"} # Output of the first component
            {"hello2": "Hello, Hello, world!"} # Output of the second component
            {"hello2": "Hello, Hello, world!"} # Final output of the pipeline
        """
        pipeline_running(self)  # type: ignore

        # Reset the visits count for each component
        self._init_graph()

        # TODO: Remove this warmup once we can check reliably whether a component has been warmed up or not
        # As of now it's here to make sure we don't have failing tests that assume warm_up() is called in run()
        self.warm_up()

        # normalize `data`
        data = self._prepare_component_input_data(data)

        # Raise if input is malformed in some way
        self._validate_input(data)

        # Initialize the inputs state
        components_inputs: Dict[str, Dict[str, Any]] = self._init_inputs_state(data)

        # Take all components that:
        # - have no inputs
        # - receive input from the user
        # - have at least one input not connected
        # - have at least one input that is variadic
        run_queue: List[Tuple[str, Component]] = self._init_run_queue(data)

        # These variables are used to detect when we're stuck in a loop.
        # Stuck loops can happen when one or more components are waiting for input but
        # no other component is going to run.
        # This can happen when a whole branch of the graph is skipped for example.
        # When we find that two consecutive iterations of the loop where the waiting_queue is the same,
        # we know we're stuck in a loop and we can't make any progress.
        #
        # They track the previous two states of the waiting_queue. So if waiting_queue would n,
        # before_last_waiting_queue would be n-2 and last_waiting_queue would be n-1.
        # When we run a component, we reset both.
        before_last_waiting_queue: Optional[Set[str]] = None
        last_waiting_queue: Optional[Set[str]] = None

        # The waiting_for_input list is used to keep track of components that are waiting for input.
        waiting_queue: List[Tuple[str, Component]] = []

        # This is what we'll return at the end
        final_outputs: Dict[Any, Any] = {}

        with tracing.tracer.trace(
            "haystack.pipeline.run",
            tags={
                "haystack.pipeline.input_data": data,
                "haystack.pipeline.output_data": final_outputs,
                "haystack.pipeline.metadata": self.metadata,
                "haystack.pipeline.max_runs_per_component": self._max_runs_per_component,
            },
        ):
            while len(run_queue) > 0:
                name, comp = run_queue.pop(0)

                if _is_lazy_variadic(comp) and not all(
                    _is_lazy_variadic(comp) for _, comp in run_queue
                ):
                    # We run Components with lazy variadic inputs only if there only Components with
                    # lazy variadic inputs left to run
                    _enqueue_waiting_component((name, comp), waiting_queue)
                    continue

                if self._component_has_enough_inputs_to_run(name, components_inputs):
                    if self.graph.nodes[name]["visits"] > self._max_runs_per_component:
                        msg = f"Maximum run count {self._max_runs_per_component} reached for component '{name}'"
                        raise PipelineMaxComponentRuns(msg)

                    res = await self._run_component(name, components_inputs[name])
                    yield {name: deepcopy(res)}

                    # Reset the waiting for input previous states, we managed to run a component
                    before_last_waiting_queue = None
                    last_waiting_queue = None

                    # We manage to run this component that was in the waiting list, we can remove it.
                    # This happens when a component was put in the waiting list but we reached it from another edge.
                    _dequeue_waiting_component((name, comp), waiting_queue)

                    for pair in self._find_components_that_will_receive_no_input(
                        name, res, components_inputs
                    ):
                        _dequeue_component(pair, run_queue, waiting_queue)
                    res = self._distribute_output(
                        name, res, components_inputs, run_queue, waiting_queue
                    )

                    if len(res) > 0:
                        final_outputs[name] = res
                else:
                    # This component doesn't have enough inputs so we can't run it yet
                    _enqueue_waiting_component((name, comp), waiting_queue)

                if len(run_queue) == 0 and len(waiting_queue) > 0:
                    # Check if we're stuck in a loop.
                    # It's important to check whether previous waitings are None as it could be that no
                    # Component has actually been run yet.
                    if (
                        before_last_waiting_queue is not None
                        and last_waiting_queue is not None
                        and before_last_waiting_queue == last_waiting_queue
                    ):
                        if self._is_stuck_in_a_loop(waiting_queue):
                            # We're stuck! We can't make any progress.
                            msg = (
                                "Pipeline is stuck running in a loop. Partial outputs will be returned. "
                                "Check the Pipeline graph for possible issues."
                            )
                            warn(RuntimeWarning(msg))
                            break

                        (name, comp) = (
                            self._find_next_runnable_lazy_variadic_or_default_component(
                                waiting_queue
                            )
                        )
                        _add_missing_input_defaults(name, comp, components_inputs)
                        _enqueue_component((name, comp), run_queue, waiting_queue)
                        continue

                    before_last_waiting_queue = (
                        last_waiting_queue.copy()
                        if last_waiting_queue is not None
                        else None
                    )
                    last_waiting_queue = {item[0] for item in waiting_queue}

                    (name, comp) = self._find_next_runnable_component(
                        components_inputs, waiting_queue
                    )
                    _add_missing_input_defaults(name, comp, components_inputs)
                    _enqueue_component((name, comp), run_queue, waiting_queue)

            yield final_outputs


async def run_async_pipeline(
    pipeline: AsyncPipeline,
    data: Dict[str, Any],
    include_outputs_from: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Helper function to run an asynchronous pipeline and return the final outputs and specific intermediate outputs.

    For fine-grained control over the intermediate outputs, use the `AsyncPipeline.run()`
    method directly and consume the generator.

    :param pipeline:
        Asynchronous pipeline to run.
    :param data:
        Input data for the pipeline.
    :param include_outputs_from:
        Set of component names whose individual outputs are to be
        included in the pipeline's output. For components that are
        invoked multiple times (in a loop), only the last-produced
        output is included.
    :returns:
        A dictionary where each entry corresponds to a component name
        and its output. If `include_outputs_from` is `None`, this dictionary
        will only contain the outputs of leaf components, i.e., components
        without outgoing connections.
    """
    outputs = [x async for x in pipeline.run(data)]

    intermediate_outputs = {
        k: v
        for d in outputs[:-1]
        for k, v in d.items()
        if include_outputs_from is None or k in include_outputs_from
    }
    final_output = outputs[-1]

    # Same logic used in the sync pipeline to accumulate extra outputs.
    for name, output in intermediate_outputs.items():
        inner = final_output.get(name)
        if inner is None:
            final_output[name] = output
        else:
            for k, v in output.items():
                if k not in inner:
                    inner[k] = v

    return final_output
