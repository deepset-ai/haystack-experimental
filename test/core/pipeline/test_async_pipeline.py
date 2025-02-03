from typing import Dict

import asyncio
import time

from haystack import component

from haystack_experimental import AsyncPipeline

@component
class Waiter:
    @component.output_types(waited_for=int)
    def run(self, wait_for: int) -> Dict[str, int]:
        time.sleep(wait_for)
        return {'waited_for': wait_for}

    @component.output_types(waited_for=int)
    async def run_async(self, wait_for: int) -> Dict[str, int]:
        await asyncio.sleep(wait_for)
        return {'waited_for': wait_for}


def test_async_pipeline_reentrance(spying_tracer):
    pp = AsyncPipeline()
    pp.add_component("wait", Waiter())

    run_data = [
        {"wait_for": 1},
        {"wait_for": 2},
    ]

    async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(async_loop)

    async def run_all():
        # Create concurrent tasks for each pipeline run
        tasks = [pp.run_async(data) for data in run_data]
        await asyncio.gather(*tasks)

    try:
        async_loop.run_until_complete(run_all())
        component_spans = [sp for sp in spying_tracer.spans if sp.operation_name == "haystack.component.run_async"]
        for span in component_spans:
            assert span.tags["haystack.component.visits"] == 1
    finally:
        async_loop.close()

