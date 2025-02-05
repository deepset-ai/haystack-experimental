# New and Old Pipeline Run Logic

We provide a way to generate traces and running times for both pipeline run logics to help you analyse any impactful 
changes between them. We provide two scripts:

- `generate_traces.py` to generate traces for both old and new pipeline run logic
- `compare_traces.py` to compare pipeline traces running order and each component inputs and outputs

## Setup

In a Python virtual environment, install both the new and old pipeline.run() logic

- `git clone -b fix/pipeline_run https://github.com/deepset-ai/haystack-experimental`
- `pip install haystack-ai`

## Generating the traces

Make sure to:

- Define any environment variables that your pipeline needs, e.g.: `OPENAI_API_KEY`
- Any (custom) component your pipeline uses must be available in `PYTHONPATH`

The `generate_traces.py` requires two arguments:

- `--pipeline-file`  the location your YAML pipeline file 
- `--pipeline-data` the input data for your pipeline - a JSON string

### Example

To generate traces for a `qa_rag_pipeline.yaml`, we need the following arguments:

- `--pipeline-file qa_rag_pipeline.yaml`
- `--pipeline-data '{"text_embedder": {"text": "What does Rhodes Statue look like?"}, "prompt_builder": {"question": "What does Rhodes Statue look like?"}}`

The command with arguments:

`python generate_traces.py --pipeline-file qa_rag_pipeline.yaml --pipeline-data '{"text_embedder": {"text": "What does Rhodes Statue look like?"}, "prompt_builder": {"question": "What does Rhodes Statue look like?"}}'` 

This will generate 3 files:

- `new_pipeline.log`
- `old_pipeline.log`
- `pipeline_execution_times.log`

## Comparing the traces

You can then run the following command:

`python compare_traces.py old_pipeline.log new_pipeline.log` 

Comparing the traces can have different results:

- `The execution order is identical in both traces`
- `The execution order differs between traces`
    - `Mismatch between components outputs`
    - `Mismatch between components inputs`
    - `Mismatch between components name/type`
    - `Mismatch between components in number of visits`
