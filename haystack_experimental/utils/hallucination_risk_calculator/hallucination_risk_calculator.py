#
# Original code Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License.
# Modified by deepset, 2025.
#

import json
import math
import random
import re
import time
from typing import Optional, Sequence

from haystack.components.generators.chat.openai import OpenAIChatGenerator

from .dataclasses import Decision, ItemMetrics, OpenAIItem


# ------------------------------------------------------------------------------------
# Core math (EDFL etc., nats)
# ------------------------------------------------------------------------------------

EPS = 1e-12


def _clamp01(x: float, eps: float = EPS) -> float:
    return min(1.0 - eps, max(eps, x))


def _safe_log(x: float) -> float:
    return math.log(max(x, EPS))


def harmonic_number(n: int) -> float:
    if n < 1:
        return 0.0
    if n < 100000:
        return sum(1.0 / r for r in range(1, n + 1))
    gamma = 0.5772156649015328606
    return math.log(n) + gamma + 1.0 / (2 * n) - 1.0 / (12 * n * n)


def expected_harmonic_distance(n: int) -> float:
    return harmonic_number(n) - 1.5


def martingale_violation_bound(n: int, L: float, C: float, alpha: float = 1.0) -> float:
    if n <= 1:
        return 0.0
    if alpha == 1.0:
        return (L * C / 8.0) * (expected_harmonic_distance(n))
    elif 0.0 < alpha < 1.0:
        return (L * C / 8.0) * (n ** (1.0 - alpha)) / (1.0 - alpha)
    else:
        return L * C / 8.0


def kl_bernoulli(p: float, q: float) -> float:
    p = _clamp01(p)
    q = _clamp01(q)
    return p * (_safe_log(p) - _safe_log(q)) + (1.0 - p) * (_safe_log(1.0 - p) - _safe_log(1.0 - q))


def inv_kl_bernoulli_upper(q: float, delta: float, tol: float = 1e-12, max_iter: int = 10000) -> float:
    q = _clamp01(q)
    delta = max(0.0, float(delta))
    hi = 1.0 - EPS
    if kl_bernoulli(hi, q) <= delta:
        return hi
    lo = q
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(mid, q) <= delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return lo


def inv_kl_bernoulli_lower(q: float, delta: float, tol: float = 1e-12, max_iter: int = 10000) -> float:
    q = _clamp01(q)
    delta = max(0.0, float(delta))
    lo = EPS
    if kl_bernoulli(lo, q) <= delta:
        return lo
    hi = q
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if kl_bernoulli(mid, q) <= delta:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi


def p_max_edfl(delta: float, q: float) -> float:
    return inv_kl_bernoulli_upper(q, delta)


def p_min_edfl(delta: float, q: float) -> float:
    return inv_kl_bernoulli_lower(q, delta)


def clip_symmetric(u: float, B: float) -> float:
    if not math.isfinite(B) or B <= 0:
        return u
    return max(-B, min(B, u))


def clip_one_sided(u: float, B: float) -> float:
    if not math.isfinite(B) or B <= 0:
        return max(u, 0.0)
    return min(max(u, 0.0), B)


def q_bar(qs: Sequence[float]) -> float:
    return sum(qs) / len(qs) if qs else 0.0


def q_lo(qs: Sequence[float]) -> float:
    return min(qs) if qs else 0.0


def delta_bar_from_logs(
    logP_y: float, logS_list_y: Sequence[float], B: float = 12.0, clip_mode: str = "one-sided"
) -> float:
    diffs = [logP_y - s for s in logS_list_y]
    clipped = [clip_one_sided(u=u, B=B) if clip_mode == "one-sided" else clip_symmetric(u=u, B=B) for u in diffs]
    return sum(clipped) / len(clipped) if clipped else 0.0


def delta_bar_from_probs(P_y: float, S_list_y: Sequence[float], B: float = 12.0, clip_mode: str = "one-sided") -> float:
    logP = _safe_log(x=P_y)
    logS = [_safe_log(x=s) for s in S_list_y]
    return delta_bar_from_logs(logP_y=logP, logS_list_y=logS, B=B, clip_mode=clip_mode)


def bits_to_trust(q_conservative: float, h_star: float) -> float:
    return kl_bernoulli(1.0 - h_star, q_conservative)


def roh_upper_bound(delta_bar: float, q_avg: float) -> float:
    return 1.0 - p_max_edfl(delta_bar, q_avg)


def isr(delta_bar: float, b2t: float) -> float:
    if b2t <= 0:
        return float("inf") if delta_bar > 0 else 1.0
    return delta_bar / b2t


# ------------------------------------------------------------------------------------
# Decision & reporting
# ------------------------------------------------------------------------------------


def decision_rule(
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
        rationale=rationale
    )


# ------------------------------------------------------------------------------------
# Skeletonization (evidence erase + closed-book masking)
# ------------------------------------------------------------------------------------

_ERASE_DEFAULT_FIELDS = ["Evidence", "Context", "Citations", "References", "Notes", "Passage", "Snippet"]

_CAPITAL_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_YEAR = re.compile(r"\b(1|2)\d{3}\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"([“\"'])(.+?)\1")


def skeletonize_prompt(text: str, fields_to_erase: Optional[Sequence[str]] = None, mask_token: str = "[…]") -> str:
    fields = list(fields_to_erase) if fields_to_erase else list(_ERASE_DEFAULT_FIELDS)
    out = text
    for field in fields:
        pattern1 = re.compile(rf"({re.escape(field)}\s*:\s*)(.*)")
        out = re.sub(pattern1, rf"\1{mask_token}", out)
        pattern2 = re.compile(rf'("{re.escape(field)}"\s*:\s*")([^"]*)(")')
        out = re.sub(pattern2, rf"\1{mask_token}\3", out)
    out = re.sub(rf"(?:{re.escape(mask_token)}\s*)+", mask_token, out)
    return out


def _extract_blocks(text: str) -> list[str]:
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    return lines if len(lines) >= 2 else [text]


def permute_prompt_blocks(text: str, seed: int) -> str:
    rng = random.Random(seed)
    blocks = _extract_blocks(text)
    idx = list(range(len(blocks)))
    rng.shuffle(idx)
    return "\n".join([blocks[i] for i in idx])


def mask_entities_numbers(text: str, strength: float, rng: random.Random, mask_token: str = "[…]") -> str:
    """
    Heuristic closed-book masker: with probability 'strength', mask each match:
      - Multi-word Capitalized sequences (names, orgs)
      - Years and numbers
      - Quoted spans
    """

    def mask_matches(pattern: re.Pattern, s: str) -> str:
        def repl(m):
            return mask_token if rng.random() < strength else m.group(0)

        return pattern.sub(repl, s)

    s = text
    # Order: quoted → capitalized entities → years → numbers
    s = mask_matches(_QUOTED, s)
    s = mask_matches(_CAPITAL_SEQ, s)
    s = mask_matches(_YEAR, s)
    s = mask_matches(_NUMBER, s)
    return s


def make_skeletons_closed_book(
    text: str,
    m: int,
    seeds: Sequence[int],
    mask_levels: Sequence[float] = (0.25, 0.35, 0.5, 0.65, 0.8, 0.9),
    mask_token: str = "[…]",
    preserve_roles: bool = True,
) -> list[str]:
    if len(seeds) < m:
        raise ValueError("Provide at least m seeds.")
    levels = list(mask_levels)
    if len(levels) < m:
        # cycle levels if fewer than m
        times = (m + len(levels) - 1) // len(levels)
        levels = (levels * times)[:m]

    ensemble = []
    for k in range(m):
        rng = random.Random(int(seeds[k]))
        lvl = float(levels[k])
        masked = mask_entities_numbers(text, lvl, rng, mask_token=mask_token)
        perm = permute_prompt_blocks(masked, seed=int(seeds[k]))
        if preserve_roles:
            lines = text.splitlines()
            if len(lines) >= 2:
                head = lines[0]
                tail = "\n".join(lines[1:])
                tail_perm = permute_prompt_blocks(
                    mask_entities_numbers(tail, lvl, rng, mask_token=mask_token), seed=int(seeds[k])
                )
                ensemble.append(head + "\n" + tail_perm)
                continue
        ensemble.append(perm)
    return ensemble


def make_skeletons_evidence_erase(
    text: str,
    m: int,
    seeds: Sequence[int],
    fields_to_erase: Optional[Sequence[str]] = None,
    mask_token: str = "[…]",
    preserve_roles: bool = True,
) -> list[str]:
    base = skeletonize_prompt(text, fields_to_erase=fields_to_erase, mask_token=mask_token)
    out = []
    for k in range(m):
        s = int(seeds[k])
        perm = permute_prompt_blocks(base, seed=s)
        if preserve_roles:
            lines = base.splitlines()
            if len(lines) >= 2:
                head = lines[0]
                tail = "\n".join(lines[1:])
                tail_perm = permute_prompt_blocks(tail, seed=s)
                out.append(head + "\n" + tail_perm)
                continue
        out.append(perm)
    return out


def make_skeleton_ensemble_auto(
    text: str,
    m: int = 6,
    seeds: Optional[Sequence[int]] = None,
    fields_to_erase: Optional[Sequence[str]] = None,
    mask_token: str = "[…]",
    skeleton_policy: str = "auto",
) -> list[str]:
    seeds = list(seeds) if seeds is not None else list(range(m))

    # Auto policy: if explicit evidence fields present or configured → evidence-erase;
    # otherwise closed-book masking.
    if skeleton_policy == "evidence_erase":
        return make_skeletons_evidence_erase(
            text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token
        )
    if skeleton_policy == "closed_book":
        return make_skeletons_closed_book(text, m=m, seeds=seeds, mask_token=mask_token)
    # auto detection
    evidence_fields = set([*(fields_to_erase or []), *_ERASE_DEFAULT_FIELDS])
    if any((f + ":") in text for f in evidence_fields):
        return make_skeletons_evidence_erase(
            text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token
        )
    return make_skeletons_closed_book(text, m=m, seeds=seeds, mask_token=mask_token)


# ------------------------------------------------------------------------------------
# OpenAI-driven signals
# ------------------------------------------------------------------------------------

_DECISION_ALLOWED = ("answer", "refuse")


def decision_messages_closed_book(user_prompt: str) -> list[dict]:
    system = (
        "You are a safety-critical QA assistant operating **without external evidence**. "
        "Decide whether to answer based on your pretrained knowledge and the prompt alone. "
        "If you are not reasonably confident, refuse. Respond with JSON only."
    )
    user = user_prompt.strip() + '\n\nReturn only JSON: {"decision":"answer"} or {"decision":"refuse"}.'
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def decision_messages_evidence(user_prompt: str) -> list[dict]:
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


def estimate_event_signals_sampling(
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
    msgs = decision_messages_closed_book(prompt) if closed_book else decision_messages_evidence(prompt)
    params = dict(model=backend.model, messages=msgs, max_tokens=max_tokens, temperature=temperature, n=n_samples)
    choices = backend.client.chat.completions.create(**params).choices
    post_decisions = [_parse_decision(c.message.content or "") for c in choices]
    if sleep_between > 0:
        time.sleep(sleep_between)
    y_label = post_decisions[0] if post_decisions else "refuse"
    P_y = sum(1 for d in post_decisions if d == y_label) / max(1, len(post_decisions))

    # Priors across skeletons
    S_list_y: list[float] = []
    q_list: list[float] = []
    for sk in skeletons:
        msgs_k = decision_messages_closed_book(sk) if closed_book else decision_messages_evidence(sk)
        params = dict(model=backend.model, messages=msgs_k, max_tokens=max_tokens, temperature=temperature, n=n_samples)
        choices_k = backend.client.chat.completions.create(**params).choices
        dec_k = [_parse_decision(c.message.content or "") for c in choices_k]
        if sleep_between > 0:
            time.sleep(sleep_between)
        qk = sum(1 for d in dec_k if d == "answer") / max(1, len(dec_k))
        syk = sum(1 for d in dec_k if d == y_label) / max(1, len(dec_k))
        q_list.append(qk)
        S_list_y.append(syk)

    return P_y, S_list_y, q_list, y_label


# ------------------------------------------------------------------------------------
# Planner
# ------------------------------------------------------------------------------------


class OpenAIPlanner:
    def __init__(
        self,
        backend: OpenAIChatGenerator,
        temperature: float = 0.5,
        max_tokens_decision: int = 8,
        q_floor: Optional[float] = None,  # prior floor to stabilize closed-book
    ) -> None:
        self.backend = backend
        self.temperature = float(temperature)
        self.max_tokens_decision = int(max_tokens_decision)
        self.q_floor = q_floor  # if None, per-item Laplace floor applied

    def _build_skeletons(self, item: OpenAIItem) -> tuple[list[str], bool]:
        seeds = item.seeds if item.seeds is not None else list(range(item.m))
        if item.skeleton_policy == "evidence_erase":
            return make_skeletons_evidence_erase(
                item.prompt, m=item.m, seeds=seeds, fields_to_erase=item.fields_to_erase
            ), False
        if item.skeleton_policy == "closed_book":
            return make_skeletons_closed_book(item.prompt, m=item.m, seeds=seeds), True
        # auto
        sk = make_skeleton_ensemble_auto(
            item.prompt, m=item.m, seeds=seeds, fields_to_erase=item.fields_to_erase, skeleton_policy="auto"
        )
        # detect whether auto chose closed_book (no explicit evidence fields present)
        closed_book = not any((f + ":") in item.prompt for f in (item.fields_to_erase or _ERASE_DEFAULT_FIELDS))
        return sk, closed_book

    def evaluate_item(
        self,
        idx: int,
        item: OpenAIItem,
        h_star: float,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
        B_clip: float = 12.0,
        clip_mode: str = "one-sided",
    ) -> ItemMetrics:
        skeletons, closed_book = self._build_skeletons(item)

        P_y, S_list_y, q_list, y_label = estimate_event_signals_sampling(
            backend=self.backend,
            prompt=item.prompt,
            skeletons=skeletons,
            n_samples=item.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens_decision,
            closed_book=closed_book,
        )

        qavg = q_bar(q_list)
        qcons = q_lo(q_list)
        # Apply prior floor (Laplace or user-provided) to avoid q_lo→0 collapse
        floor = self.q_floor if self.q_floor is not None else 1.0 / (item.n_samples + 2)
        qcons = max(qcons, floor)

        dbar = delta_bar_from_probs(P_y, S_list_y, B=B_clip, clip_mode=clip_mode)
        dec = decision_rule(
            dbar,
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
        h_star: float = 0.05,
        isr_threshold: float = 1.0,
        margin_extra_bits: float = 0.0,
        B_clip: float = 12.0,
        clip_mode: str = "one-sided",
    ) -> list[ItemMetrics]:
        return [
            self.evaluate_item(
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
