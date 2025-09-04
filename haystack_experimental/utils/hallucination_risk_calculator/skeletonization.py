# Original code Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License (see LICENSE-MIT).
# Modified by deepset, 2025.
# Licensed under the Apache License, Version 2.0 (see LICENSE-APACHE).
import random
import re
from typing import Optional, Sequence

_ERASE_DEFAULT_FIELDS = ["Evidence", "Context", "Citations", "References", "Notes", "Passage", "Snippet"]

_CAPITAL_SEQ = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_YEAR = re.compile(r"\b(1|2)\d{3}\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_QUOTED = re.compile(r"([“\"'])(.+?)\1")


def _skeletonize_prompt(text: str, fields_to_erase: Optional[Sequence[str]] = None, mask_token: str = "[…]") -> str:
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


def _permute_prompt_blocks(text: str, seed: int) -> str:
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


def _make_skeletons_closed_book(
    *,
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
        perm = _permute_prompt_blocks(masked, seed=int(seeds[k]))
        if preserve_roles:
            lines = text.splitlines()
            if len(lines) >= 2:
                head = lines[0]
                tail = "\n".join(lines[1:])
                tail_perm = _permute_prompt_blocks(
                    mask_entities_numbers(tail, lvl, rng, mask_token=mask_token), seed=int(seeds[k])
                )
                ensemble.append(head + "\n" + tail_perm)
                continue
        ensemble.append(perm)
    return ensemble


def _make_skeletons_evidence_erase(
    *,
    text: str,
    m: int,
    seeds: Sequence[int],
    fields_to_erase: Optional[Sequence[str]] = None,
    mask_token: str = "[…]",
    preserve_roles: bool = True,
) -> list[str]:
    base = _skeletonize_prompt(text, fields_to_erase=fields_to_erase, mask_token=mask_token)
    out = []
    for k in range(m):
        s = int(seeds[k])
        perm = _permute_prompt_blocks(base, seed=s)
        if preserve_roles:
            lines = base.splitlines()
            if len(lines) >= 2:
                head = lines[0]
                tail = "\n".join(lines[1:])
                tail_perm = _permute_prompt_blocks(tail, seed=s)
                out.append(head + "\n" + tail_perm)
                continue
        out.append(perm)
    return out


def _make_skeleton_ensemble_auto(
    *,
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
        return _make_skeletons_evidence_erase(
            text=text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token
        )
    if skeleton_policy == "closed_book":
        return _make_skeletons_closed_book(text=text, m=m, seeds=seeds, mask_token=mask_token)
    # auto detection
    evidence_fields: set[str] = set(*(fields_to_erase or []), *_ERASE_DEFAULT_FIELDS)
    if any((f + ":") in text for f in evidence_fields):
        return _make_skeletons_evidence_erase(
            text=text, m=m, seeds=seeds, fields_to_erase=fields_to_erase, mask_token=mask_token
        )
    return _make_skeletons_closed_book(text=text, m=m, seeds=seeds, mask_token=mask_token)
