from typing import Dict, Any

from haystack import Pipeline, component
from haystack.core.serialization import default_from_dict, default_to_dict

from haystack_experimental.core.super_component.base import SuperComponentBase

@component
class SuperComponent(SuperComponentBase):
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the pipeline wrapper to a dictionary representation.

        :return: Dictionary containing serialized pipeline wrapper data
        """
        serialized_pipeline = self.pipeline.to_dict()
        return default_to_dict(
            self, pipeline=serialized_pipeline, input_mapping=self.input_mapping, output_mapping=self.output_mapping
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuperComponent":
        """
        Create a pipeline wrapper instance from a dictionary representation.

        :param data: Dictionary containing serialized pipeline wrapper data
        :return: New PipelineWrapper instance
        """
        pipeline = Pipeline.from_dict(data["init_parameters"]["pipeline"])
        data["init_parameters"]["pipeline"] = pipeline

        return default_from_dict(cls, data)