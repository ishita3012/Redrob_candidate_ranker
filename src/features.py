"""
Per-candidate FIT features, scored against a JDSpec (never against hardcoded
constants). Each feature returns a 0..1 value plus short human-readable evidence
strings used later for reasoning.

Design choices grounded in the empirical analysis of this pool:
  * Skills lists are uniform noise (~12K occurrences each) -> they only *corroborate*,
    never drive a score. The trustworthy content signal is career-history DESCRIPTIONS.
  * "Product vs services" matters to the JD repeatedly -> we measure the fraction of
    the career spent at non-consulting (product) companies, not a binary.
  * Experience is a soft band preference, not a hard filter (JD: "a range, not a
    requirement"), so out-of-band decays smoothly.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

from jd_spec import JDSpec
from gates import core_evidence_hits, evidence_hits, _is_consulting, _title, _all_titles


def _descs(candidate: Dict[str, Any]) -> str:
    return " ".join(j.get("description", "") for j in candidate.get("career_history", [])).lower()


# ---------------------------------------------------------------------------
# 1. Title fit  (current + history alignment with the JD's role titles)
# ---------------------------------------------------------------------------
def f_title(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    cur = _title(candidate)
    reasons: List[str] = []
    score = 0.0
    if any(k in cur for k in spec.role_titles):
        score = 1.0
        reasons.append(f"{candidate['profile']['current_title']} (target-role title)")
    elif any(k in cur for k in spec.adjacent_titles):
        score = 0.55
        reasons.append(f"{candidate['profile']['current_title']} (adjacent title)")
    # seniority nudge (JD wants senior judgement, not juniors)
    if any(s in cur for s in ["senior", "staff", "principal", "lead"]):
        score = min(1.0, score + 0.1)
    elif "junior" in cur or "associate" in cur:
        score *= 0.7
        reasons.append("junior-level title")
    # past ML roles add a little even if current title is adjacent
    past = sum(1 for t in _all_titles(candidate)[1:] if any(k in t for k in spec.role_titles))
    if past and score < 1.0:
        score = min(1.0, score + 0.15)
        reasons.append(f"{past} prior ML-role(s)")
    return score, reasons


# ---------------------------------------------------------------------------
# 2. Evidence depth  (domain-specific proof of work in descriptions)
# ---------------------------------------------------------------------------
def f_evidence(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    core = core_evidence_hits(candidate, spec)
    allh = evidence_hits(candidate, spec)
    supporting = [h for h in allh if h not in core]
    # core hits dominate; supporting (production/scale/python) corroborates
    score = min(1.0, 0.30 * len(core) + 0.05 * len(supporting))
    reasons = []
    if core:
        reasons.append("built: " + ", ".join(sorted(set(core))[:4]))
    if any(s in _descs(candidate) for s in ["production", "deployed", "real users",
                                            "at scale", "millions", "serving"]):
        reasons.append("production/scale evidence")
    return score, reasons


# ---------------------------------------------------------------------------
# 3. Product-vs-services  (fraction of career at non-consulting companies)
# ---------------------------------------------------------------------------
def f_product(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    if "only_consulting" not in spec.disqualifier_concepts:
        return 0.5, []  # JD doesn't care; neutral
    hist = candidate.get("career_history", [])
    if not hist:
        return 0.5, []
    months = [(j.get("duration_months", 0), j.get("company", "")) for j in hist]
    total = sum(m for m, _ in months) or 1
    product = sum(m for m, c in months if not _is_consulting(c))
    frac = product / total
    reasons = []
    if frac >= 0.8:
        reasons.append("product-company career")
    elif frac <= 0.2:
        reasons.append("mostly services/consulting")
    return frac, reasons


# ---------------------------------------------------------------------------
# 4. Experience-band fit  (soft, decays outside the JD band)
# ---------------------------------------------------------------------------
def f_experience(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    lo, hi = spec.exp_band
    ideal = spec.exp_ideal or (lo, hi)
    ilo, ihi = ideal
    reasons = [f"{yoe:.1f} yrs experience"]
    if ilo <= yoe <= ihi:
        return 1.0, reasons
    if lo <= yoe <= hi:
        return 0.85, reasons
    # smooth decay outside the hard band (1 point per year, floor 0.2)
    dist = (lo - yoe) if yoe < lo else (yoe - hi)
    return max(0.2, 0.85 - 0.15 * dist), reasons


# ---------------------------------------------------------------------------
# 5. Location fit  (JD names preferred cities + relocation policy)
# ---------------------------------------------------------------------------
def f_location(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    if not spec.locations and not spec.location_country:
        return 0.5, []
    loc = candidate.get("profile", {}).get("location", "").lower()
    country = candidate.get("profile", {}).get("country", "").lower()
    relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)
    if any(c in loc for c in spec.locations):
        return 1.0, ["in a preferred city"]
    if spec.location_country and spec.location_country in country:
        return 0.8, []  # in-country, relocatable
    if relocate:
        return 0.6, ["willing to relocate"]
    return 0.25, ["outside preferred geography, no relocation"]  # JD: no visa sponsorship


# ---------------------------------------------------------------------------
# 6. Stability  (tenure pattern; the JD rejects title-chasers who switch ~1.5y)
# ---------------------------------------------------------------------------
def f_stability(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    hist = candidate.get("career_history", [])
    if len(hist) < 2:
        return 0.6, []  # too little history to judge
    durations = [j.get("duration_months", 0) for j in hist]
    avg = sum(durations) / len(durations)
    if len(hist) >= 3 and avg < 18:
        return 0.2, [f"frequent short stints (avg {avg:.0f}mo)"]
    if avg >= 30:
        return 1.0, ["stable tenure"]
    if avg >= 20:
        return 0.8, []
    return 0.5, []


# ---------------------------------------------------------------------------
# 7. Skill corroboration  (weak signal only — skills lists are noisy)
# ---------------------------------------------------------------------------
def f_skill_corroboration(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[float, List[str]]:
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    # only platform-verified assessment scores count (resist self-reported noise)
    relevant = [v for k, v in assessments.items()
                if any(t in k.lower() for t in ["ml", "machine", "nlp", "python",
                                                "deep", "data", "search", "rank"])]
    if not relevant:
        return 0.4, []
    avg = sum(relevant) / len(relevant) / 100.0
    reasons = ["verified assessments"] if avg >= 0.7 else []
    return avg, reasons


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
FEATURE_FUNCS = {
    "title": f_title,
    "evidence": f_evidence,
    "product": f_product,
    "experience": f_experience,
    "stability": f_stability,
    "location": f_location,
    "skill_corroboration": f_skill_corroboration,
}


def compute_features(candidate: Dict[str, Any], spec: JDSpec
                     ) -> Dict[str, Tuple[float, List[str]]]:
    return {name: fn(candidate, spec) for name, fn in FEATURE_FUNCS.items()}
