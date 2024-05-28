from collections import defaultdict

from haystack import Pipeline, component


@component
class PipelineWrapper:
    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline_instance = pipeline
        self.pipeline = pipeline.to_dict()

        # This component has the same inputs as the wrapped pipeline. The wrapped pipeline might have
        # a component expecting multiple inputs like this:
        #
        # {
        #     'llm': {
        #         'prompt': {'type': ..., 'is_mandatory': True},
        #         'generation_kwargs': {'type': ..., 'is_mandatory': False, 'default_value': None}
        #     }
        # }
        #
        # In turn, this wrapper components would have nested inputs:
        #
        # {
        #   "this_component": {
        #     'llm': {
        #       'prompt': {'type': ..., 'is_mandatory': True},
        #       'generation_kwargs': {'type': ..., 'is_mandatory': False, 'default_value': None}
        #     }
        #   }
        # }
        #
        # This component would be difficult to connect, and to avoid nesting the inputs we flatten the wrapped
        # inputs using this naming convention:
        #
        # <this component input> -> <wrapped_component_name>:<wrapped_input_name>
        #
        # the inputs of this component would then be:
        # {
        #   'llm:prompt': {...},
        #   'llm.generation_kwargs': {...}
        # }
        for component_name, inputs in self._pipeline_instance.inputs().items():
            for input_name, typedef in inputs.items():
                call_args = [self, f"{component_name}:{input_name}", typedef["type"]]
                if "default" in typedef:
                    call_args.append(typedef["default_value"])
                component.set_input_type(*call_args)

        # Same logic for the output
        for component_name, outputs in self._pipeline_instance.outputs().items():
            kwargs = {}
            for output_name, typedef in outputs.items():
                kwargs[f"{component_name}:{output_name}"] = typedef
            component.set_output_types(self, **kwargs)

    def run(self, **kwargs):
        # split the inputs
        inner_data = defaultdict(dict)
        for name, value in kwargs.items():
            component_name, input_name = name.split(":")
            inner_data[component_name][input_name] = value

        return self._pipeline_instance.run(data=inner_data)
