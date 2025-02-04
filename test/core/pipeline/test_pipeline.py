from concurrent.futures import ThreadPoolExecutor
from haystack_experimental import Pipeline


def test_pipeline_thread_safety(waiting_component, spying_tracer):
    # Initialize pipeline with synchronous components
    pp = Pipeline()
    pp.add_component("wait", waiting_component())

    run_data = [
        {"wait_for": 1},
        {"wait_for": 2},
    ]

    # Use ThreadPoolExecutor to run pipeline calls in parallel
    with ThreadPoolExecutor(max_workers=len(run_data)) as executor:
        # Submit pipeline runs to the executor
        futures = [
            executor.submit(pp.run, data)
            for data in run_data
        ]

        # Wait for all futures to complete
        for future in futures:
            future.result()

    # Verify component visits using tracer
    component_spans = [
        sp for sp in spying_tracer.spans
        if sp.operation_name == "haystack.component.run"
    ]

    for span in component_spans:
        assert span.tags["haystack.component.visits"] == 1