# Original code Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License (see LICENSE-MIT).
# Modified by deepset, 2025.
# Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE).
import json
import re
import time
from typing import Any, Optional, Sequence, Union

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from .core_math import bits_to_trust, delta_bar_from_probs, isr, q_bar, q_lo, roh_upper_bound
from .dataclasses import Decision, HallucinationScoreConfig, ItemMetrics, OpenAIItem
from .skeletonization import (
    _ERASE_DEFAULT_FIELDS,
    _make_skeleton_ensemble_auto,
    _make_skeletons_closed_book,
    _make_skeletons_evidence_erase,
)

_DECISION_ALLOWED = ("answer", "refuse")


def _decision_messages_closed_book(
    user_prompt: str,
) -> list[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]]:
    system = (
        "You are a safety-critical QA assistant operating **without external evidence**. "
        "Decide whether to answer based on your pretrained knowledge and the prompt alone. "
        "If you are not reasonably confident, refuse. Respond with JSON only."
    )
    user = user_prompt.strip() + '\n\nReturn only JSON: {"decision":"answer"} or {"decision":"refuse"}.'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _decision_messages_evidence(
    user_prompt: str,
) -> list[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam]]:
    system = (
        "You are a safety-critical QA assistant. Decide whether to answer based on the "
        "provided prompt and its internal evidence/context. If evidence is insufficient or "
        "conflicting, refuse. Respond with JSON only."
    )
    user = user_prompt.strip() + '\n\nReturn only JSON: {"decision":"answer"} or {"decision":"refuse"}.'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _parse_decision(text: str) -> str:
    """
    Parse model output to extract decision: "answer" or "refuse".

    Defaults to "refuse" if parsing fails or invalid.

    :param text: Model output text.
    """
    try:
        obj = json.loads(text)
        d = str(obj.get("decision", "")).strip().lower()
        if d in _DECISION_ALLOWED:
            return d
    except Exception:
        pass
    m = re.search(r'{"\s*decision\s*"\s*:\s*"(answer|refuse)"}', text, flags=re.I)
    if m:
        return m.group(1).lower()
    low = text.strip().lower()
    if "refuse" in low and "answer" not in low:
        return "refuse"
    if "answer" in low and "refuse" not in low:
        return "answer"
    return "refuse"


def _estimate_event_signals_sampling(
    *,
    backend: OpenAIChatGenerator,
    prompt: str,
    skeletons: list[str],
    n_samples: int = 3,
    temperature: float = 0.5,
    max_tokens: int = 8,
    closed_book: bool = True,
    sleep_between: float = 0.0,
) -> tuple[float, list[float], list[float], str]:
    # Posterior (full prompt)
    msgs = _decision_messages_closed_book(prompt) if closed_book else _decision_messages_evidence(prompt)
    choices = backend.client.chat.completions.create(
        model=backend.model, messages=msgs, max_tokens=max_tokens, temperature=temperature, n=n_samples
    ).choices
    post_decisions = [_parse_decision(text=c.message.content or "") for c in choices]
    if sleep_between > 0:
        time.sleep(sleep_between)
    y_label = post_decisions[0] if post_decisions else "refuse"
    P_y = sum(1 for d in post_decisions if d == y_label) / max(1, len(post_decisions))

    # Priors across skeletons
    S_list_y: list[float] = []
    q_list: list[float] = []
    for sk in skeletons:
        msgs_k = _decision_messages_closed_book(sk) if closed_book else _decision_messages_evidence(sk)
        choices_k = backend.client.chat.completions.create(
            model=backend.model, messages=msgs_k, max_tokens=max_tokens, temperature=temperature, n=n_samples
        ).choices
        dec_k = [_parse_decision(text=c.message.content or "") for c in choices_k]
        if sleep_between > 0:
            time.sleep(sleep_between)
        qk = sum(1 for d in dec_k if d == "answer") / max(1, len(dec_k))
        syk = sum(1 for d in dec_k if d == y_label) / max(1, len(dec_k))
        q_list.append(qk)
        S_list_y.append(syk)

    return P_y, S_list_y, q_list, y_label


def decision_rule(
    *,
    delta_bar: float,
    q_conservative: float,
    q_avg: float,
    h_star: float,
    isr_threshold: float = 1.0,
    margin_extra_bits: float = 0.0,
) -> Decision:
    """
    Decision rule: answer if ISR >= threshold and Δ̄ >= B2T + margin.

    :param delta_bar: EDFL Δ̄ (nats)
    :param q_conservative: conservative prior q_lo (0-1)
    :param q_avg: average prior q̄ (0-1)
    :param h_star: target hallucination rate (0-1)
    :param isr_threshold: ISR threshold (default 1.0)
    :param margin_extra_bits: extra bits margin (default 0.0)

    :returns:
        Decision dataclass
    """
    b2t_val = bits_to_trust(q_conservative, h_star)
    isr_val = isr(delta_bar, b2t_val)
    roh = roh_upper_bound(delta_bar, q_avg)
    rationale = (
        f"Δ̄={delta_bar:.4f} nats, B2T={b2t_val:.4f}, ISR={isr_val:.3f} "
        f"(thr={isr_threshold:.3f}), extra_bits={margin_extra_bits:.3f}; "
        f"EDFL RoH bound={roh:.3f}"
    )
    return Decision(
        answer=(isr_val >= isr_threshold) and (delta_bar >= b2t_val + max(0.0, margin_extra_bits)),
        isr=isr_val,
        b2t=b2t_val,
        roh_bound=roh,
        margin=margin_extra_bits,
        threshold=isr_threshold,
        rationale=rationale,
    )


class OpenAIPlanner:
    def __init__(
        self,
        backend: OpenAIChatGenerator,
        temperature: float = 0.5,
        max_tokens_decision: int = 8,
        q_floor: Optional[float] = None,
    ) -> None:
        """
        Initialize OpenAIPlanner with given parameters.

        :param backend: OpenAIChatGenerator instance for API calls.
        :param temperature: Sampling temperature for decision calls.
        :param max_tokens_decision: Max tokens for decision response.
        :param q_floor: Optional prior floor (0-1) to avoid q_lo→0 collapse.
                        If None, per-item Laplace floor 1/(n_samples+2) applied.
        """
        self.backend = backend
        self.temperature = float(temperature)
        self.max_tokens_decision = int(max_tokens_decision)
        self.q_floor = q_floor

    def _build_skeletons(self, item: OpenAIItem) -> tuple[list[str], bool]:
        seeds = item.seeds if item.seeds is not None else list(range(item.m))
        if item.skeleton_policy == "evidence_erase":
            return _make_skeletons_evidence_erase(
                text=item.prompt, m=item.m, seeds=seeds, fields_to_erase=item.fields_to_erase
            ), False
        if item.skeleton_policy == "closed_book":
            return _make_skeletons_closed_book(text=item.prompt, m=item.m, seeds=seeds), True
        # auto
        sk = _make_skeleton_ensemble_auto(
            text=item.prompt, m=item.m, seeds=seeds, fields_to_erase=item.fields_to_erase, skeleton_policy="auto"
        )
        # detect whether auto chose closed_book (no explicit evidence fields present)
        closed_book = not any((f + ":") in item.prompt for f in (item.fields_to_erase or _ERASE_DEFAULT_FIELDS))
        return sk, closed_book

    def _evaluate_item(
        self,
        *,
        idx: int,
        item: OpenAIItem,
        h_star: float,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
        B_clip: float = 12.0,
        clip_mode: str = "one-sided",
    ) -> ItemMetrics:
        skeletons, closed_book = self._build_skeletons(item=item)

        P_y, S_list_y, q_list, y_label = _estimate_event_signals_sampling(
            backend=self.backend,
            prompt=item.prompt,
            skeletons=skeletons,
            closed_book=closed_book,
            n_samples=item.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens_decision,
        )

        qavg = q_bar(q_list)
        qcons = q_lo(q_list)
        # Apply prior floor (Laplace or user-provided) to avoid q_lo→0 collapse
        floor = self.q_floor if self.q_floor is not None else 1.0 / (item.n_samples + 2)
        qcons = max(qcons, floor)

        dbar = delta_bar_from_probs(P_y, S_list_y, B=B_clip, clip_mode=clip_mode)
        dec = decision_rule(
            delta_bar=dbar,
            q_conservative=qcons,
            q_avg=qavg,
            h_star=h_star,
            isr_threshold=isr_threshold,
            margin_extra_bits=margin_extra_bits,
        )

        meta = {
            "q_list": q_list,
            "S_list_y": S_list_y,
            "P_y": P_y,
            "closed_book": closed_book,
            "q_floor": floor,
            "y_label": y_label,
        }

        return ItemMetrics(
            item_id=idx,
            delta_bar=dbar,
            q_avg=qavg,
            q_conservative=qcons,
            b2t=dec.b2t,
            isr=dec.isr,
            roh_bound=dec.roh_bound,
            decision_answer=dec.answer,
            rationale=dec.rationale + f"; y='{y_label}'",
            attempted=item.attempted,
            answered_correctly=item.answered_correctly,
            meta=meta,
        )

    def run(
        self,
        items: Sequence[OpenAIItem],
        *,
        h_star: float = 0.05,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
        B_clip: float = 12.0,
        clip_mode: str = "one-sided",
    ) -> list[ItemMetrics]:
        """
        Evaluate a sequence of OpenAIItem instances.
        """
        return [
            self._evaluate_item(
                idx=i,
                item=it,
                h_star=h_star,
                isr_threshold=isr_threshold,
                margin_extra_bits=margin_extra_bits,
                B_clip=B_clip,
                clip_mode=clip_mode,
            )
            for i, it in enumerate(items)
        ]


def calculate_hallucination_metrics(
    prompt: str,
    hallucination_score_config: HallucinationScoreConfig,
    chat_generator: OpenAIChatGenerator,
) -> dict[str, Any]:
    """
    Calculate hallucination metrics for a given prompt using the OpenAIPlanner.
    """
    item = OpenAIItem(
        prompt=prompt,
        n_samples=hallucination_score_config.n_samples,
        m=hallucination_score_config.m,
        skeleton_policy=hallucination_score_config.skeleton_policy,
    )
    planner = OpenAIPlanner(
        chat_generator,
        temperature=hallucination_score_config.temperature,
    )
    metrics = planner.run(
        [item],
        h_star=hallucination_score_config.h_star,
        isr_threshold=hallucination_score_config.isr_threshold,
        margin_extra_bits=hallucination_score_config.margin_extra_bits,
        B_clip=hallucination_score_config.B_clip,
        clip_mode=hallucination_score_config.clip_mode,
    )
    hallucination_meta = {
        "hallucination_decision": "ANSWER" if metrics[0].decision_answer else "REFUSE",
        "hallucination_risk": metrics[0].roh_bound,
        "hallucination_rationale": metrics[0].rationale,
    }
    return hallucination_meta
