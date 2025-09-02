#
# Original code Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License.
# Modified by deepset, 2025.
#
from dataclasses import dataclass
from typing import Literal, Optional


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
    :param isr: The Inverse Scaling Rate (ISR) associated with the decision.
    :param b2t: The bits-to-threshold (B2T) value associated with the decision.
    :param roh_bound: The risk of hallucination (ROH) bound associated with the decision.
    :param margin: The margin of safety in bits for the decision.
    :param threshold: The threshold value in bits for making the decision.
    :param rationale: A string providing the rationale behind the decision.
    """


@dataclass
class OpenAIItem:
    prompt: str
    n_samples: int = 3
    m: int = 6
    seeds: Optional[list[int]] = None
    fields_to_erase: Optional[list[str]] = None  # evidence-based mode
    mask_token: str = "[…]"
    skeleton_policy: Literal["auto", "evidence_erase", "closed_book"] = "auto"
    attempted: Optional[bool] = None
    answered_correctly: Optional[bool] = None
    meta: Optional[dict] = None


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
    attempted: Optional[bool] = None
    answered_correctly: Optional[bool] = None
    meta: Optional[dict] = None
