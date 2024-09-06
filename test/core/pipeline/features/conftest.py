from dataclasses import dataclass, field
from typing import Generator, Tuple, List, Dict, Any, Set, Union
from pathlib import Path
import re
import asyncio

import pytest
from pytest_bdd import when, then, parsers

from haystack_experimental.core import AsyncPipeline, run_async_pipeline
import contextlib
import dataclasses
import uuid
from typing import Dict, Any, Optional, List, Iterator

from haystack.tracing import Span, Tracer, enable_tracing, disable_tracing


PIPELINE_NAME_REGEX = re.compile(r"\[(.*)\]")


@dataclasses.dataclass
class SpyingSpan(Span):
    operation_name: str
    tags: Dict[str, Any] = dataclasses.field(default_factory=dict)

    trace_id: Optional[str] = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())
    )
    span_id: Optional[str] = dataclasses.field(
        default_factory=lambda: str(uuid.uuid4())
    )

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def get_correlation_data_for_logs(self) -> Dict[str, Any]:
        return {"trace_id": self.trace_id, "span_id": self.span_id}


class SpyingTracer(Tracer):
    def current_span(self) -> Optional[Span]:
        return self.spans[-1] if self.spans else None

    def __init__(self) -> None:
        self.spans: List[SpyingSpan] = []

    @contextlib.contextmanager
    def trace(
        self, operation_name: str, tags: Optional[Dict[str, Any]] = None
    ) -> Iterator[Span]:
        new_span = SpyingSpan(operation_name)

        for key, value in (tags or {}).items():
            new_span.set_tag(key, value)

        self.spans.append(new_span)

        yield new_span


@pytest.fixture()
def spying_tracer() -> Generator[SpyingTracer, None, None]:
    tracer = SpyingTracer()
    enable_tracing(tracer)

    yield tracer

    # Make sure to disable tracing after the test to avoid affecting other tests
    disable_tracing()


@dataclass
class PipelineRunData:
    """
    Holds the inputs and expected outputs for a single Pipeline run.
    """

    inputs: Dict[str, Any]
    include_outputs_from: Set[str] = field(default_factory=set)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    expected_run_order: List[str] = field(default_factory=list)


@dataclass
class _PipelineResult:
    """
    Holds the outputs and the run order of a single Pipeline run.
    """

    outputs: Dict[str, Any]
    run_order: List[str]


@when("I run the Pipeline", target_fixture="pipeline_result")
def run_pipeline(
    pipeline_data: Tuple[AsyncPipeline, List[PipelineRunData]], spying_tracer
) -> Union[List[Tuple[_PipelineResult, PipelineRunData]], Exception]:
    """
    Attempts to run a pipeline with the given inputs.
    `pipeline_data` is a tuple that must contain:
    * A Pipeline instance
    * The data to run the pipeline with

    If successful returns a tuple of the run outputs and the expected outputs.
    In case an exceptions is raised returns that.
    """
    pipeline, pipeline_run_data = pipeline_data[0], pipeline_data[1]

    results: List[_PipelineResult] = []

    async def run_inner(data, include_outputs_from):
        return await run_async_pipeline(pipeline, data.inputs, include_outputs_from)

    for data in pipeline_run_data:
        try:
            async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(async_loop)
            outputs = async_loop.run_until_complete(
                run_inner(data, data.include_outputs_from)
            )

            run_order = [
                span.tags["haystack.component.name"]
                for span in spying_tracer.spans
                if "haystack.component.name" in span.tags
            ]
            results.append(_PipelineResult(outputs=outputs, run_order=run_order))
            spying_tracer.spans.clear()
        except Exception as e:
            return e
        finally:
            async_loop.close()
    return [e for e in zip(results, pipeline_run_data)]


@then("draw it to file")
def draw_pipeline(pipeline_data: Tuple[AsyncPipeline, List[PipelineRunData]], request):
    """
    Draw the pipeline to a file with the same name as the test.
    """
    if m := PIPELINE_NAME_REGEX.search(request.node.name):
        name = m.group(1).replace(" ", "_")
        pipeline = pipeline_data[0]
        graphs_dir = Path(request.config.rootpath) / "test_pipeline_graphs"
        graphs_dir.mkdir(exist_ok=True)
        pipeline.draw(graphs_dir / f"{name}.png")


@then("it should return the expected result")
def check_pipeline_result(
    pipeline_result: List[Tuple[_PipelineResult, PipelineRunData]]
):
    for res, data in pipeline_result:
        assert res.outputs == data.expected_outputs


@then("components ran in the expected order")
def check_pipeline_run_order(
    pipeline_result: List[Tuple[_PipelineResult, PipelineRunData]]
):
    for res, data in pipeline_result:
        assert res.run_order == data.expected_run_order


@pytest.mark.asyncio
@then(parsers.parse("it must have raised {exception_class_name}"))
def check_pipeline_raised(pipeline_result: Exception, exception_class_name: str):
    assert pipeline_result.__class__.__name__ == exception_class_name
