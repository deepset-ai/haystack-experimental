# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

from haystack.dataclasses import ChatMessage

from haystack_experimental.components.generators.chat.openai import HallucinationScoreConfig, OpenAIChatGenerator

# os.environ["OPENAI_API_KEY"] = ""

llm = OpenAIChatGenerator(model="gpt-4o")

# Closed Book Example
closed_book_result = llm.run(
    messages=[ChatMessage.from_user(text="Who won the 2019 Nobel Prize in Physics?")],
    hallucination_score_config=HallucinationScoreConfig(skeleton_policy="closed_book"),
)
print(f"Decision: {closed_book_result['replies'][0].meta['hallucination_decision']}")
print(f"Risk bound: {closed_book_result['replies'][0].meta['hallucination_risk']:.3f}")
print(f"Risk bound: {closed_book_result['replies'][0].meta['hallucination_rationale']}")
print(f"Answer:\n{closed_book_result['replies'][0].text}")
print("---\n")

# Evidence-based Example
rag_result = llm.run(
    messages=[
        ChatMessage.from_user(
            text="Task: Answer strictly based on the evidence provided below.\n"
            "Question: Who won the Nobel Prize in Physics in 2019?\n"
            "Evidence:\n"
            "- Nobel Prize press release (2019): James Peebles (1/2); Michel Mayor & Didier Queloz (1/2).\n"
            "Constraints: If evidence is insufficient or conflicting, refuse."
        )
    ],
    hallucination_score_config=HallucinationScoreConfig(skeleton_policy="evidence_erase"),
)
print(f"Decision: {rag_result['replies'][0].meta['hallucination_decision']}")
print(f"Risk bound: {rag_result['replies'][0].meta['hallucination_risk']:.3f}")
print(f"Rationale: {rag_result['replies'][0].meta['hallucination_rationale']}")
print(f"Answer:\n{closed_book_result['replies'][0].text}")
print("---")
