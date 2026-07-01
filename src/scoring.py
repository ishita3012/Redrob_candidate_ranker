"""
Stage-2 rerank scoring — a two-factor model taken straight from the JD's own framing:

    score = relevance × reachability

The JD says a candidate can be a great fit yet, "for hiring purposes, not actually
available" — so we separate *how well they fit the role* (relevance) from *whether we
can actually engage and hire them* (reachability). Traps (honeypots, keyword-stuffers)
are removed upstream by the Stage-1 gates, not modelled here.

  relevance    = trust × Σ(feature × weight) × (1 − red_flag_penalty)
                 trust discounts profiles with consistency concerns; the JD's explicit
                 negatives (consulting-only, title-chasing, research-only) are red flags
                 AND are also reflected directly in features (product, stability, evidence).
  reachability = a bounded behavioral factor calibrated to the pool's measured
                 percentiles: response rate, recency, open-to-work, notice, interviews.

Weights reflect the empirical finding that career-description EVIDENCE and title carry
the signal, while skills lists are noise.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from datetime import date, datetime

from jd_spec import JDSpec
from features import compute_features
from gates import detect_disqualifiers, honeypot_suspicion

REFERENCE_DATE = date(2026, 6, 21)

# Relevance feature weights (sum to 1.0).
FIT_WEIGHTS: Dict[str, float] = {
    "evidence": 0.28,
    "title": 0.20,
    "semantic": 0.17,
    "product": 0.11,
    "experience": 0.09,
    "stability": 0.07,
    "location": 0.05,
    "skill_corroboration": 0.03,
}
assert abs(sum(FIT_WEIGHTS.values()) - 1.0) < 1e-9


def _days_since(s: str) -> int:
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return 9999


# ---------------------------------------------------------------------------
# reachability  (behavioral: can we actually engage and hire them?)
# bounded ~[0.5, 1.10] so it modifies relevance rather than dominating it.
# ---------------------------------------------------------------------------
def reachability(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    if not spec.behavioral_priorities:
        return 1.0, []
    s = candidate.get("redrob_signals", {})
    r = 1.0
    reasons: List[str] = []

    rr = s.get("recruiter_response_rate", 0) or 0
    if rr >= 0.62:
        r += 0.06; reasons.append(f"{rr:.0%} response rate")
    elif rr < 0.25:
        r -= 0.18; reasons.append(f"low {rr:.0%} response rate")

    di = _days_since(s.get("last_active_date", ""))
    if di <= 45:
        r += 0.05; reasons.append("recently active")
    elif di > 180:
        r -= 0.20; reasons.append(f"inactive {di}d")

    if s.get("open_to_work_flag"):
        r += 0.05; reasons.append("open to work")

    notice = s.get("notice_period_days", 90)
    pref = spec.notice_pref_days or 30
    if notice <= pref:
        r += 0.05; reasons.append(f"{notice}d notice")
    elif notice >= 120:
        r -= 0.08; reasons.append(f"long {notice}d notice")

    ic = s.get("interview_completion_rate", 0) or 0
    if ic >= 0.8:
        r += 0.03
    elif ic < 0.4:
        r -= 0.08; reasons.append("low interview completion")

    return max(0.5, min(1.10, r)), reasons


# ---------------------------------------------------------------------------
# relevance  (trust-and-flag-adjusted fit to the JD)
# ---------------------------------------------------------------------------
def relevance(candidate: Dict[str, Any], spec: JDSpec, semantic_score: float
              ) -> Tuple[float, Dict[str, float], Dict[str, List[str]], List[str], float]:
    feats = compute_features(candidate, spec)
    values = {k: v for k, (v, _) in feats.items()}
    reasons = {k: r for k, (_, r) in feats.items()}
    values["semantic"] = semantic_score
    reasons["semantic"] = []

    fit = sum(FIT_WEIGHTS[k] * values.get(k, 0.0) for k in FIT_WEIGHTS)
    trust = 1.0 - 0.5 * honeypot_suspicion(candidate)          # profile consistency
    flags, red_flag_penalty = detect_disqualifiers(candidate, spec)  # JD's explicit negatives

    rel = fit * trust * (1.0 - red_flag_penalty)
    return rel, values, reasons, flags, trust


def compute_relevance(candidate: Dict[str, Any], spec: JDSpec,
                      semantic_score: float = 0.5) -> Dict[str, Any]:
    rel, values, reasons, flags, trust = relevance(candidate, spec, semantic_score)
    reach, reach_reasons = reachability(candidate, spec)
    score = rel * reach

    return {
        "score": score,
        "composite": score,          # legacy alias
        "relevance": rel,
        "reachability": reach,
        "availability": reach,       # legacy alias
        "fit": sum(FIT_WEIGHTS[k] * values.get(k, 0.0) for k in FIT_WEIGHTS),
        "trust": trust,
        "authenticity": trust,       # legacy alias
        "disqualifier_penalty": detect_disqualifiers(candidate, spec)[1],
        "feature_values": values,
        "feature_reasons": reasons,
        "reachability_reasons": reach_reasons,
        "availability_reasons": reach_reasons,  # legacy alias
        "disqualifiers": flags,
    }
