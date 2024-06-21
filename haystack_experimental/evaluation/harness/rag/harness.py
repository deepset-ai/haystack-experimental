# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict, List, Optional, Set

from haystack import Pipeline
from haystack.evaluation.eval_run_result import EvaluationRunResult

from ...util.helpers import (
    aggregate_batched_pipeline_outputs,
    deaggregate_batched_pipeline_inputs,
)
from ...util.pipeline_pair import PipelinePair
from ..evalution_harness import EvaluationHarness
from .evaluation_pipeline import default_rag_evaluation_pipeline
from .parameters import (
    RAGEvaluationInput,
    RAGEvaluationMetric,
    RAGEvaluationOutput,
    RAGEvaluationOverrides,
    RAGExpectedComponent,
    RAGExpectedComponentMetadata,
)


class RAGEvaluationHarness(
    EvaluationHarness[RAGEvaluationInput, RAGEvaluationOverrides, RAGEvaluationOutput]
):
    """
    Evaluation harness for evaluating RAG pipelines.
    """

    def __init__(
        self,
        rag_pipeline: Pipeline,
        rag_components: Dict[RAGExpectedComponent, RAGExpectedComponentMetadata],
        metrics: Set[RAGEvaluationMetric],
    ):
        """
        Create an evaluation harness for evaluating basic RAG pipelines.

        :param rag_pipeline:
            The RAG pipeline to evaluate.
        :param rag_components:
            A mapping of expected components to their metadata.
        :param metrics:
            The metrics to use during evaluation.
        """
        super().__init__()

        self._validate_rag_components(rag_pipeline, rag_components)

        self.rag_pipeline = rag_pipeline
        self.rag_components = rag_components
        self.metrics = metrics
        self.evaluation_pipeline = default_rag_evaluation_pipeline(metrics)

    @classmethod
    def default_with_embedding_retriever(
        cls, rag_pipeline: Pipeline, metrics: Set[RAGEvaluationMetric]
    ) -> "RAGEvaluationHarness":
        """
        Create a default evaluation harness for evaluating RAG pipelines with a query embedder.

        :param rag_pipeline:
            The RAG pipeline to evaluate. The following assumptions are made:
            - The query embedder component is named 'query_embedder' and has a 'text' input.
            - The document retriever component is named 'retriever' and has a 'documents' output.
            - The response generator component is named 'generator' and has a 'replies' output.
        :param metrics:
            The metrics to use during evaluation.
        """
        rag_components = {
            RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                name="query_embedder", input_mapping={"query": "text"}
            ),
            RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                name="retriever", output_mapping={"retrieved_documents": "documents"}
            ),
            RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
                name="generator", output_mapping={"replies": "replies"}
            ),
        }

        return cls(rag_pipeline, rag_components, deepcopy(metrics))

    @classmethod
    def default_with_keyword_retriever(
        cls, rag_pipeline: Pipeline, metrics: Set[RAGEvaluationMetric]
    ) -> "RAGEvaluationHarness":
        """
        Create a default evaluation harness for evaluating RAG pipelines with a keyword retriever.

        :param rag_pipeline:
            The RAG pipeline to evaluate. The following assumptions are made:
            - The document retriever component is named 'retriever' and has a 'query' input and a 'documents' output.
            - The response generator component is named 'generator' and has a 'replies' output.
        :param metrics:
            The metrics to use during evaluation.
        """
        rag_components = {
            RAGExpectedComponent.QUERY_PROCESSOR: RAGExpectedComponentMetadata(
                name="retriever", input_mapping={"query": "query"}
            ),
            RAGExpectedComponent.DOCUMENT_RETRIEVER: RAGExpectedComponentMetadata(
                name="retriever", output_mapping={"retrieved_documents": "documents"}
            ),
            RAGExpectedComponent.RESPONSE_GENERATOR: RAGExpectedComponentMetadata(
                name="generator", output_mapping={"replies": "replies"}
            ),
        }

        return cls(rag_pipeline, rag_components, deepcopy(metrics))

    def run(  # noqa: D102
        self,
        inputs: RAGEvaluationInput,
        *,
        overrides: Optional[RAGEvaluationOverrides] = None,
        run_name: Optional[str] = "RAG Evaluation",
    ) -> RAGEvaluationOutput:
        rag_inputs = self._prepare_rag_pipeline_inputs(inputs)
        eval_inputs = self._prepare_eval_pipeline_additional_inputs(inputs)
        pipeline_pair = self._generate_eval_run_pipelines(overrides)

        pipeline_outputs = pipeline_pair.run_first_as_batch(rag_inputs, eval_inputs)
        rag_outputs, eval_outputs = (
            pipeline_outputs["first"],
            pipeline_outputs["second"],
        )

        result_inputs = {
            "questions": inputs.queries,
            "contexts": [
                [doc.content for doc in docs]
                for docs in self._lookup_component_output(
                    RAGExpectedComponent.DOCUMENT_RETRIEVER,
                    rag_outputs,
                    "retrieved_documents",
                )
            ],
            "responses": self._lookup_component_output(
                RAGExpectedComponent.RESPONSE_GENERATOR, rag_outputs, "replies"
            ),
        }
        if inputs.ground_truth_answers is not None:
            result_inputs["ground_truth_answers"] = inputs.ground_truth_answers
        if inputs.ground_truth_documents is not None:
            result_inputs["ground_truth_documents"] = [
                [doc.content for doc in docs] for docs in inputs.ground_truth_documents
            ]

        assert run_name is not None
        run_results = EvaluationRunResult(
            run_name,
            inputs=result_inputs,
            results=eval_outputs,
        )

        return RAGEvaluationOutput(
            evaluated_pipeline=pipeline_pair.first.dumps(),
            evaluation_pipeline=pipeline_pair.second.dumps(),
            inputs=deepcopy(inputs),
            results=run_results,
        )

    def _lookup_component_output(
        self,
        component: RAGExpectedComponent,
        outputs: Dict[str, Dict[str, Any]],
        output_name: str,
    ) -> Any:
        name = self.rag_components[component].name
        mapping = self.rag_components[component].output_mapping
        output_name = mapping[output_name]
        return outputs[name][output_name]

    def _generate_eval_run_pipelines(
        self, overrides: Optional[RAGEvaluationOverrides]
    ) -> PipelinePair:
        if overrides is None:
            rag_overrides = None
            eval_overrides = None
        else:
            rag_overrides = overrides.rag_pipeline
            eval_overrides = overrides.eval_pipeline

        if eval_overrides is not None:
            for metric in eval_overrides.keys():
                if metric not in self.metrics:
                    raise ValueError(
                        f"Cannot override parameters of unused evaluation metric '{metric.value}'"
                    )

            eval_overrides = {k.value: v for k, v in eval_overrides.items()}  # type: ignore

        rag_pipeline = self._override_pipeline(self.rag_pipeline, rag_overrides)
        eval_pipeline = self._override_pipeline(self.evaluation_pipeline, eval_overrides)  # type: ignore

        return PipelinePair(
            first=rag_pipeline,
            second=eval_pipeline,
            outputs_to_inputs=self._map_rag_eval_pipeline_io(),
            map_first_outputs=lambda x: self._aggregate_rag_outputs(  # pylint: disable=unnecessary-lambda
                x
            ),
            included_first_outputs={
                self.rag_components[RAGExpectedComponent.DOCUMENT_RETRIEVER].name,
                self.rag_components[RAGExpectedComponent.RESPONSE_GENERATOR].name,
            },
        )

    def _aggregate_rag_outputs(
        self, outputs: List[Dict[str, Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        aggregate = aggregate_batched_pipeline_outputs(outputs)

        # We only care about the first response from the generator.
        generator_name = self.rag_components[
            RAGExpectedComponent.RESPONSE_GENERATOR
        ].name
        replies_output_name = self.rag_components[
            RAGExpectedComponent.RESPONSE_GENERATOR
        ].output_mapping["replies"]
        aggregate[generator_name][replies_output_name] = [
            r[0] for r in aggregate[generator_name][replies_output_name]
        ]

        return aggregate

    def _map_rag_eval_pipeline_io(self) -> Dict[str, List[str]]:
        # We currently only have metric components in the eval pipeline.
        # So, we just map those inputs to the outputs of the rag pipeline.
        metric_inputs_to_component_outputs = {
            RAGEvaluationMetric.DOCUMENT_MAP: {
                "retrieved_documents": (
                    RAGExpectedComponent.DOCUMENT_RETRIEVER,
                    "retrieved_documents",
                )
            },
            RAGEvaluationMetric.DOCUMENT_MRR: {
                "retrieved_documents": (
                    RAGExpectedComponent.DOCUMENT_RETRIEVER,
                    "retrieved_documents",
                )
            },
            RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT: {
                "retrieved_documents": (
                    RAGExpectedComponent.DOCUMENT_RETRIEVER,
                    "retrieved_documents",
                )
            },
            RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT: {
                "retrieved_documents": (
                    RAGExpectedComponent.DOCUMENT_RETRIEVER,
                    "retrieved_documents",
                )
            },
            RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY: {
                "predicted_answers": (
                    RAGExpectedComponent.RESPONSE_GENERATOR,
                    "replies",
                )
            },
            RAGEvaluationMetric.ANSWER_FAITHFULNESS: {
                "contexts": (
                    RAGExpectedComponent.DOCUMENT_RETRIEVER,
                    "retrieved_documents",
                ),
                "predicted_answers": (
                    RAGExpectedComponent.RESPONSE_GENERATOR,
                    "replies",
                ),
            },
            RAGEvaluationMetric.CONTEXT_RELEVANCE: {
                "contexts": (
                    RAGExpectedComponent.DOCUMENT_RETRIEVER,
                    "retrieved_documents",
                ),
            },
        }

        outputs_to_inputs: Dict[str, List[str]] = {}
        for metric in self.metrics:
            io = metric_inputs_to_component_outputs[metric]
            for metric_input_name, (component, component_output_name) in io.items():
                component_out = (
                    f"{self.rag_components[component].name}."
                    f"{self.rag_components[component].output_mapping[component_output_name]}"
                )
                metric_in = f"{metric.value}.{metric_input_name}"
                if component_out not in outputs_to_inputs:
                    outputs_to_inputs[component_out] = []
                outputs_to_inputs[component_out].append(metric_in)

        return outputs_to_inputs

    def _prepare_rag_pipeline_inputs(
        self, inputs: RAGEvaluationInput
    ) -> List[Dict[str, Dict[str, Any]]]:
        query_embedder_name = self.rag_components[
            RAGExpectedComponent.QUERY_PROCESSOR
        ].name
        query_embedder_text_input = self.rag_components[
            RAGExpectedComponent.QUERY_PROCESSOR
        ].input_mapping["query"]

        if inputs.additional_rag_inputs is not None:
            # Ensure that the query embedder input is not provided as additional input.
            existing = inputs.additional_rag_inputs.get(query_embedder_name)
            if existing is not None:
                existing = existing.get(query_embedder_text_input)  # type: ignore
                if existing is not None:
                    raise ValueError(
                        f"Query embedder input '{query_embedder_text_input}' cannot be provided as additional input."
                    )

            # Add the queries as an aggregate input.
            rag_inputs = deepcopy(inputs.additional_rag_inputs)
            if query_embedder_name not in rag_inputs:
                rag_inputs[query_embedder_name] = {}
            rag_inputs[query_embedder_name][query_embedder_text_input] = deepcopy(
                inputs.queries
            )
        else:
            rag_inputs = {
                query_embedder_name: {
                    query_embedder_text_input: deepcopy(inputs.queries)
                }
            }

        separate_rag_inputs = deaggregate_batched_pipeline_inputs(rag_inputs)
        return separate_rag_inputs

    def _prepare_eval_pipeline_additional_inputs(
        self, inputs: RAGEvaluationInput
    ) -> Dict[str, Dict[str, Any]]:
        eval_inputs: Dict[str, Dict[str, List[Any]]] = {}

        for metric in self.metrics:
            if metric in (
                RAGEvaluationMetric.DOCUMENT_MAP,
                RAGEvaluationMetric.DOCUMENT_MRR,
                RAGEvaluationMetric.DOCUMENT_RECALL_SINGLE_HIT,
                RAGEvaluationMetric.DOCUMENT_RECALL_MULTI_HIT,
            ):
                if inputs.ground_truth_documents is None:
                    raise ValueError(
                        f"Ground truth documents required for metric '{metric.value}'."
                    )
                if len(inputs.ground_truth_documents) != len(inputs.queries):
                    raise ValueError(
                        "Length of ground truth documents should match the number of queries."
                    )

                eval_inputs[metric.value] = {
                    "ground_truth_documents": inputs.ground_truth_documents
                }
            elif metric in (
                RAGEvaluationMetric.ANSWER_FAITHFULNESS,
                RAGEvaluationMetric.CONTEXT_RELEVANCE,
            ):
                eval_inputs[metric.value] = {"questions": inputs.queries}
            elif metric == RAGEvaluationMetric.SEMANTIC_ANSWER_SIMILARITY:
                if inputs.ground_truth_answers is None:
                    raise ValueError(
                        f"Ground truth answers required for metric '{metric.value}'."
                    )
                if len(inputs.ground_truth_answers) != len(inputs.queries):
                    raise ValueError(
                        "Length of ground truth answers should match the number of queries."
                    )

                eval_inputs[metric.value] = {
                    "ground_truth_answers": inputs.ground_truth_answers
                }

        return eval_inputs

    @staticmethod
    def _validate_rag_components(
        pipeline: Pipeline,
        components: Dict[RAGExpectedComponent, RAGExpectedComponentMetadata],
    ):
        for e in RAGExpectedComponent:
            if e not in components:
                raise ValueError(
                    f"RAG evaluation harness requires metadata for the '{e.value}' component."
                )

        pipeline_outputs = pipeline.outputs(
            include_components_with_connected_outputs=True
        )
        pipeline_inputs = pipeline.inputs(include_components_with_connected_inputs=True)

        for component, metadata in components.items():
            if (
                metadata.name not in pipeline_outputs
                or metadata.name not in pipeline_inputs
            ):
                raise ValueError(
                    f"Expected '{component.value}' component named '{metadata.name}' not found in pipeline."
                )

            comp_inputs = pipeline_inputs[metadata.name]
            comp_outputs = pipeline_outputs[metadata.name]

            for needle in metadata.input_mapping.values():
                if needle not in comp_inputs:
                    raise ValueError(
                        f"Required input '{needle}' not found in '{component.value}' "
                        f"component named '{metadata.name}'."
                    )

            for needle in metadata.output_mapping.values():
                if needle not in comp_outputs:
                    raise ValueError(
                        f"Required output '{needle}' not found in '{component.value}' "
                        f"component named '{metadata.name}'."
                    )
