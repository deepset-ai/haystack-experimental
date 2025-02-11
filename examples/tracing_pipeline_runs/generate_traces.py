import argparse
import json
import logging
import time

from haystack import Pipeline as OldPipeline
from haystack import tracing

from custom_tracer import LoggingTracer
from haystack_experimental.core.pipeline.pipeline import Pipeline as NewPipeline

logging.basicConfig(filename='old_pipeline.log', format="%(levelname)s - %(name)s -  %(message)s")

tracing.tracer.is_content_tracing_enabled = True # set to "True" to enable tracing/logging content (inputs/outputs)
tracing.enable_tracing(LoggingTracer()) # enable the custom tracer

def change_log_file(logger, new_log_file):
    """
    Use to rotate the log file.
    """
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

    new_file_handler = logging.FileHandler(new_log_file)
    formatter = logging.Formatter("%(levelname)s - %(name)s -  %(message)s")
    new_file_handler.setFormatter(formatter)
    logger.addHandler(new_file_handler)

def run_pipeline(pipeline_run, yam_pipeline_file_location, new=True):
    """
    Runs a pipeline loaded from yaml.
    """
    with open(yam_pipeline_file_location, "r") as f_in:
        if new:
            pipeline = NewPipeline.loads(f_in)
        else:
            pipeline = OldPipeline.loads(f_in)

    start_time = time.time()
    _ = pipeline.run(pipeline_run)
    end_time = time.time()

    execution_time = end_time - start_time
    pipeline_type = "NewPipeline" if new else "OldPipeline"

    with open("pipeline_execution_times.log", "a") as log_file:
        log_file.write(f"{pipeline_type} execution time: {execution_time:.4f} seconds\n")

def main():
    """
    Main execution routine to generate pipeline traces.
    """
    parser = argparse.ArgumentParser(description='Run pipeline comparison with configurable inputs')
    parser.add_argument('--pipeline-file', type=str, required=True,
                      help='Path to the YAML pipeline file')
    parser.add_argument('--pipeline-data', type=str, required=True,
                      help='JSON string containing the pipeline run data')
    
    args = parser.parse_args()
    
    # Parse the JSON string into a dictionary
    try:
        pipeline_run_data = json.loads(args.pipeline_data)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format for pipeline-data")

    run_pipeline(pipeline_run_data, args.pipeline_file, new=False)
    change_log_file(logging.getLogger(), "new_pipeline.log")    # change the logging file
    run_pipeline(pipeline_run_data, args.pipeline_file, new=True)


if __name__ == '__main__':
    main()