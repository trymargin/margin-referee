"""The switch gate's statistics: a paired comparison with a best-of-N correction.

Ported from Margin's hosted product (``src/margin/autoroute.py`` and ``src/margin/fit_check.py``);
a test in the product repo runs both on the same inputs and fails if they ever disagree.

The question it answers: could the candidate be WORSE than the model you run today by more than
the bar allows (3% of your current pass rate), given the cases both ran? Cases both passed, or
both failed, say nothing about the gap; only the cases they disagree on do (McNemar). Two floors
keep a small panel from clearing on almost nothing: the half-width never falls below the
"rule of three" (3/n), and the panel must be at least as large as an all-passing absolute test
would need. Trying several candidates and keeping the best raises the bar (Bonferroni), because
the best of several looks better than it is.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from statistics import NormalDist
from typing import Iterable, Optional

#: The bar: the candidate must hold at least 97% of the current model's pass rate.
PARITY_THRESHOLD = 0.97
#: One-sided 95% normal quantile.
EVIDENCE_Z = 1.645
_ZERO_EVENT_BOUND = 3.0
DEFAULT_BASELINE_RATE = 0.9


def family_z(candidates: int, base_z: float = EVIDENCE_Z) -> float:
    """The z that holds 95% confidence across ``candidates`` tried (Bonferroni)."""
    if candidates < 1:
        raise ValueError(f"candidates must be >= 1, got {candidates}")
    if candidates == 1:
        return base_z
    norm = NormalDist()
    return norm.inv_cdf(1.0 - (1.0 - norm.cdf(base_z)) / candidates)


def trials_needed(threshold: float = PARITY_THRESHOLD, z: float = EVIDENCE_Z) -> int:
    """Fewest all-passing trials whose Wilson lower bound clears ``threshold`` (88 at 0.97)."""
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}")
    return math.ceil(z * z * threshold / (1.0 - threshold))


@dataclass(frozen=True)
class Paired:
    paired: int
    ties: int
    baseline_only: int
    candidate_only: int
    shortfall: float
    upper: float
    margin: float
    supported: bool


def paired_bound(pairs: Iterable[tuple[float, float]], *, threshold: float = PARITY_THRESHOLD,
                 z: float = EVIDENCE_Z) -> Optional[Paired]:
    """``pairs`` is one ``(baseline_score, candidate_score)`` per case, each in [0, 1]. None when
    there are no pairs: no evidence, no claim."""
    ties = baseline_only = candidate_only = 0
    baseline_passes = 0.0
    for b, c in pairs:
        baseline_passes += b
        if b > c:
            baseline_only += 1
        elif c > b:
            candidate_only += 1
        else:
            ties += 1
    paired = ties + baseline_only + candidate_only
    if paired == 0:
        return None
    discordant = baseline_only + candidate_only
    shortfall = (baseline_only - candidate_only) / paired
    half = max(z * (discordant ** 0.5) / paired, _ZERO_EVENT_BOUND / paired)
    margin = (baseline_passes / paired) * (1.0 - threshold)
    upper = shortfall + half
    return Paired(paired, ties, baseline_only, candidate_only, shortfall, upper, margin,
                  paired >= trials_needed(threshold, z) and upper <= margin)


def losses_absorbed(n: int, baseline_rate: float = DEFAULT_BASELINE_RATE,
                    candidates: int = 1) -> Optional[int]:
    """The most cases a candidate may lose (and win none) on an ``n``-case suite and still
    clear the bar; None when even a clean sweep cannot."""
    passes = round(n * baseline_rate)
    z = family_z(max(1, candidates))
    best = None
    for k in range(0, passes + 1):
        pairs = [(1.0 if i < passes else 0.0, 0.0 if i < k else (1.0 if i < passes else 0.0))
                 for i in range(n)]
        pb = paired_bound(pairs, z=z)
        if pb is None or not pb.supported:
            break
        best = k
    return best


@lru_cache(maxsize=64)
def smallest_suite(baseline_rate: float = DEFAULT_BASELINE_RATE, limit: int = 2000,
                   candidates: int = 1) -> int:
    """The fewest cases that can prove a switch at this current pass rate, with no
    disagreement (100 against a perfect current model, 111 against one passing 90%)."""
    lo, hi = trials_needed(), limit
    while lo < hi:
        mid = (lo + hi) // 2
        if losses_absorbed(mid, baseline_rate, candidates) is None:
            lo = mid + 1
        else:
            hi = mid
    return lo


def gate_floor(candidates: int = 1) -> int:
    """Cases needed to prove a switch when ``candidates`` models were tried on the same cases."""
    return max(trials_needed(z=family_z(candidates)), smallest_suite(1.0, candidates=candidates))
