# ruff: noqa: D103
# Original code Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License (see LICENSE-MIT).
# Modified by deepset, 2025.
# Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE).

import math
from typing import Sequence

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
