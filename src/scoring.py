"""
Stage-2 rerank scoring.

    composite = fit × availability × (1 − disqualifier) × authenticity

This mirrors how the JD frames the signals: fit is the relevance backbone;
behavioral availability is a *modifier* (JD: "down-weight" unavailable people, not
zero them); disqualifiers and honeypot-authenticity are gates/penalties.

Calibration is grounded in the measured distribution of THIS pool, so thresholds
are meaningful rather than arbitrary:
    recruiter_response_rate : p50 0.44, p75 0.62, p90 0.73
    days since last_active  : p50 130,  p75 187,  p90 231
    notice_period_days      : p50 90,   pref <=30
    interview_completion    : p50 0.62
The weights reflect the empirical finding that skills lists are noise and the
trustworthy content signal is career-description EVIDENCE + title.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple
from datetime import date, datetime

from jd_spec import JDSpec
from features import compute_features
from gates import detect_disqualifiers, honeypot_suspicion

REFERENCE_DATE = date(2026, 6, 21)

# Fit weights (sum to 1.0). Evidence + title dominate; skills only corroborate.
FIT_WEIGHTS: Dict[str, float] = {
    "evidence": 0.30,
    "title": 0.22,
    "semantic": 0.18,
    "product": 0.12,
    "experience": 0.10,
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
# availability multiplier  (centered ~1.0, bounded so it modifies, not dominates)
# ---------------------------------------------------------------------------
def availability_multiplier(candidate: Dict[str, Any], spec: JDSpec
                            ) -> Tuple[float, List[str]]:
    if not spec.behavioral_priorities:
        return 1.0, []
    s = candidate.get("redrob_signals", {})
    m = 1.0
    reasons: List[str] = []

    rr = s.get("recruiter_response_rate", 0) or 0
    if rr >= 0.62:
        m += 0.06; reasons.append(f"{rr:.0%} response rate")
    elif rr < 0.25:
        m -= 0.18; reasons.append(f"low {rr:.0%} response rate")

    di = _days_since(s.get("last_active_date", ""))
    if di <= 45:
        m += 0.05; reasons.append("recently active")
    elif di > 180:
        m -= 0.20; reasons.append(f"inactive {di}d")

    if s.get("open_to_work_flag"):
        m += 0.05; reasons.append("open to work")

    notice = s.get("notice_period_days", 90)
    pref = spec.notice_pref_days or 30
    if notice <= pref:
        m += 0.05; reasons.append(f"{notice}d notice")
    elif notice >= 120:
        m -= 0.08; reasons.append(f"long {notice}d notice")

    ic = s.get("interview_completion_rate", 0) or 0
    if ic >= 0.8:
        m += 0.03
    elif ic < 0.4:
        m -= 0.08; reasons.append("low interview completion")

    return max(0.5, min(1.10, m)), reasons


# ---------------------------------------------------------------------------
# fit  (weighted feature sum, with an injected semantic score)
# ---------------------------------------------------------------------------
def compute_fit(candidate: Dict[str, Any], spec: JDSpec, semantic_score: float
                ) -> Tuple[float, Dict[str, float], Dict[str, List[str]]]:
    feats = compute_features(candidate, spec)
    values = {k: v for k, (v, _) in feats.items()}
    reasons = {k: r for k, (_, r) in feats.items()}
    values["semantic"] = semantic_score
    reasons["semantic"] = []
    fit = sum(FIT_WEIGHTS[k] * values.get(k, 0.0) for k in FIT_WEIGHTS)
    return fit, values, reasons


def compute_relevance(candidate: Dict[str, Any], spec: JDSpec,
                      semantic_score: float = 0.5) -> Dict[str, Any]:
    fit, values, reasons = compute_fit(candidate, spec, semantic_score)
    avail_mult, avail_reasons = availability_multiplier(candidate, spec)
    disq_flags, disq_penalty = detect_disqualifiers(candidate, spec)
    authenticity = 1.0 - 0.5 * honeypot_suspicion(candidate)

    composite = fit * avail_mult * (1.0 - disq_penalty) * authenticity

    return {
        "composite": composite,
        "fit": fit,
        "availability": avail_mult,
        "disqualifier_penalty": disq_penalty,
        "authenticity": authenticity,
        "feature_values": values,
        "feature_reasons": reasons,
        "availability_reasons": avail_reasons,
        "disqualifiers": disq_flags,
    }
