# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

from haystack import Pipeline, component


@component
class PipelineWrapper:
    """
    PipelineWrapper wraps a pipeline into a single component.

    This component has the same inputs as the wrapped pipeline. The wrapped pipeline might have
    a component expecting multiple inputs like this:

    ```python
    {
        'llm': {
            'prompt': {'type': ..., 'is_mandatory': True},
            'generation_kwargs': {'type': ..., 'is_mandatory': False, 'default_value': None}
        }
    }
    ```

    In turn, this wrapper components would have nested inputs:

    ```python
    {
        "this_component": {
        'llm': {
            'prompt': {'type': ..., 'is_mandatory': True},
            'generation_kwargs': {'type': ..., 'is_mandatory': False, 'default_value': None}
        }
        }
    }
    ````

    This component would be difficult to connect, so we wrap the data we send to the pipeline and
    the return value of the pipeline using this convention:

    <this component input> -> <wrapped_component_name>:<wrapped_input_name>

    the inputs of this component would then be:

    ```python
    {
        'llm:prompt': {...},
        'llm.generation_kwargs': {...}
    }
    ```
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline_instance = pipeline
        self.pipeline = pipeline.to_dict()

        # Create this component's inputs to match those of the wrapped pipeline
        for component_name, inputs in self._pipeline_instance.inputs().items():
            for input_name, typedef in inputs.items():
                call_args = [self, f"{component_name}:{input_name}", typedef["type"]]
                if "default" in typedef:
                    call_args.append(typedef["default_value"])
                component.set_input_type(*call_args)

        # Create this component's outputs to match those of the wrapped pipeline
        for component_name, outputs in self._pipeline_instance.outputs().items():
            kwargs = {}
            for output_name, typedef in outputs.items():
                kwargs[f"{component_name}:{output_name}"] = typedef
            component.set_output_types(self, **kwargs)

    def run(self, **kwargs):
        """
        Before running the wrapped pipeline, we unwrap `kwargs`, so we can invoke the underlying pipeline.

        For example, if the following is passed as `kwargs` (note the "wrapping convention" of the input):

        ```python
        {'comp_name:inp_name': 'This is the value that was passed'}
        ```

        It will be converted to a "regular" pipeline input:

        ```python
        {'comp_name': {'inp_name': 'This is the value that was passed'}}
        ```
        """
        unwrapped_data = defaultdict(dict)
        for name, value in kwargs.items():
            component_name, input_name = name.split(":")
            unwrapped_data[component_name][input_name] = value

        # Run the wrapped pipeline, it will return unwrapped values
        ret = self._pipeline_instance.run(data=dict(unwrapped_data))

        # Wrap the return value of the pipeline run
        wrapped_ret = {}
        for k, v in ret.items():
            component_name = k
            for output_name, ret_value in v.items():
                wrapped_ret[f"{component_name}:{output_name}"] = ret_value

        # return the wrapped output
        return wrapped_ret
