# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest

from haystack.dataclasses import ChatMessage

from haystack_experimental.utils.hallucination_risk_calculator.dataclasses import HallucinationScoreConfig
from haystack_experimental.components.generators.chat.openai import OpenAIChatGenerator


class TestOpenAIChatGenerator:
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY", None),
        reason="Export an env var called OPENAI_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_openai_chat_generator(self):
        llm = OpenAIChatGenerator(model="gpt-4o")
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
        replies = rag_result["replies"]
        assert len(replies) == 1
        assert "hallucination_decision" in replies[0].meta
        assert "hallucination_risk" in replies[0].meta
        assert "hallucination_rationale" in replies[0].meta
