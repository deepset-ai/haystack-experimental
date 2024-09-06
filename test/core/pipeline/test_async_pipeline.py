# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pytest

from haystack import component
from haystack.testing.sample_components import AddFixedValue
from haystack_experimental.core import AsyncPipeline, run_async_pipeline


@component
class AsyncDoubleWithOriginal:
    """
    Doubles the input value and returns the original value as well.
    """

    def __init__(self) -> None:
        self.async_executed = False

    @component.output_types(value=int, original=int)
    def run(self, value: int):
        raise NotImplementedError()

    @component.output_types(value=int, original=int)
    async def async_run(self, value: int):
        self.async_executed = True
        return {"value": value * 2, "original": value}


@pytest.mark.asyncio
async def test_async_pipeline():
    pipeline = AsyncPipeline()
    pipeline.add_component("first_addition", AddFixedValue(add=2))
    pipeline.add_component("second_addition", AddFixedValue())
    pipeline.add_component("double", AsyncDoubleWithOriginal())
    pipeline.connect("first_addition", "double")
    pipeline.connect("double.value", "second_addition")

    outputs = {}
    # since enumerate doesn't work with async generators
    expected_intermediate_outputs = [
        {"first_addition": {"result": 5}},
        {"double": {"value": 10, "original": 5}},
        {"second_addition": {"result": 11}},
    ]

    outputs = [o async for o in pipeline.run({"first_addition": {"value": 3}})]
    intermediate_outputs = outputs[:-1]
    final_output = outputs[-1]

    assert expected_intermediate_outputs == intermediate_outputs
    assert final_output == {
        "double": {"original": 5},
        "second_addition": {"result": 11},
    }
    assert pipeline.get_component("double").async_executed is True
    pipeline.get_component("double").async_executed = False

    other_final_outputs = await run_async_pipeline(
        pipeline,
        {"first_addition": {"value": 3}},
        include_outputs_from={"double", "second_addition", "first_addition"},
    )
    assert other_final_outputs == {
        "first_addition": {"result": 5},
        "double": {"value": 10, "original": 5},
        "second_addition": {"result": 11},
    }
    assert pipeline.get_component("double").async_executed is True
