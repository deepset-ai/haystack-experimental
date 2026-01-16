# Original code Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License (see LICENSE-MIT).
# Modified by deepset, 2025.
# Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE).
from dataclasses import dataclass
from typing import Literal


@dataclass
class HallucinationScoreConfig:
    """
    Configuration for hallucination risk assessment using OpenAIPlanner.
    """

    skeleton_policy: Literal["auto", "evidence_erase", "closed_book"] = "closed_book"
    n_samples: int = 7
    m: int = 6
    temperature: float = 0.3
    h_star: float = 0.05
    isr_threshold: float = 1.0
    margin_extra_bits: float = 0.2
    B_clip: float = 12.0
    clip_mode: Literal["one-sided", "symmetric"] = "one-sided"


@dataclass
class Decision:
    answer: bool
    isr: float
    b2t: float
    roh_bound: float
    margin: float
    threshold: float
    rationale: str

    """
    Represents the decision made by the model regarding whether to answer a question or abstain.

    :param answer: A boolean indicating whether the model decided to answer (True) or abstain (False).
    :param isr: The Information Sufficiency Ratio (ISR) associated with the decision.
    :param b2t: The Bits-to-Trust (B2T) value associated with the decision.
    :param roh_bound: The EDFL hallucination risk bound associated with the decision.
    :param margin: The margin of safety in bits for the decision.
    :param threshold: The threshold value in bits for making the decision.
    :param rationale: A string providing the rationale behind the decision.
    """


@dataclass
class OpenAIItem:
    prompt: str
    n_samples: int = 3
    m: int = 6
    seeds: list[int] | None = None
    fields_to_erase: list[str] | None = None  # evidence-based mode
    mask_token: str = "[…]"
    skeleton_policy: Literal["auto", "evidence_erase", "closed_book"] = "auto"
    attempted: bool | None = None
    answered_correctly: bool | None = None
    meta: dict | None = None


@dataclass
class ItemMetrics:
    item_id: int
    delta_bar: float
    q_avg: float
    q_conservative: float
    b2t: float
    isr: float
    roh_bound: float
    decision_answer: bool
    rationale: str
    attempted: bool | None = None
    answered_correctly: bool | None = None
    meta: dict | None = None
