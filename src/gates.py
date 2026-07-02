"""
Stage-1 gates: honeypot detection, hard disqualifiers, and the prefilter that
turns 100K candidates into the small pool worth ranking (recall -> precision).

Calibration is grounded in the empirical distribution of THIS pool, but the rules
are stated as impossibilities / JD-derived red flags, so they generalize:

  * expert proficiency with 0 months used        -> impossible (cliff: 0, then 21 cand at >=3)
  * a skill used far longer than the whole career -> impossible (>48mo excess: 13 cand)
  * grad-year vs experience is DELIBERATELY NOT USED as a honeypot signal:
    it flags ~21,700 legitimate candidates (recent degree + senior exp) and is pure noise.

The prefilter keeps a candidate only if it is plausibly relevant to the JDSpec
(role/adjacent title OR real evidence in descriptions) and not a honeypot. On this
JD that reduces 100K -> ~1K, which is the retrieval step of a retrieve->rank system.
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple

from jd_spec import JDSpec

# Consulting-firm gazetteer (used when a JD activates the only_consulting disqualifier).
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "mindtree", "ltimindtree", "lti", "mphasis", "niit", "genpact",
    "cts", "deloitte", "ey", "pwc", "kpmg", "hexaware", "birlasoft",
}

# Titles that are clearly outside an AI/ML role (the keyword-stuffer carriers).
NON_TECH_TITLE_TOKENS = {
    "marketing", "hr ", "human resource", "accountant", "accounting", "sales",
    "content writer", "graphic", "customer support", "civil engineer",
    "mechanical engineer", "operations manager", "business analyst", "project manager",
    "recruiter", "finance", "administrative",
}

# --- Lexicons for the JD's explicitly-named disqualifiers -------------------
# "primary expertise is computer vision, speech, or robotics without NLP/IR"
CV_SPEECH_ROBOTICS = ("computer vision", "image classification", "object detection",
    "opencv", "image segmentation", "speech recognition", "asr", "tts", "robotics",
    "slam", "autonomous driving", "face recognition", "video analytics", "pose estimation")
NLP_IR_TERMS = ("nlp", "natural language", "retrieval", "search", "ranking", "recommendation",
    "llm", "language model", "embedding", "information retrieval", "semantic", "text ")
# "senior engineer who hasn't written production code in the last 18 months"
LEADERSHIP_TITLES = ("head of", "director", "vp ", "vice president", "chief", "cto",
    "engineering manager", "delivery manager", "general manager", "principal architect",
    "solution architect", "enterprise architect")
HANDS_ON_VERBS = ("implemented", "built", "wrote", "coded", "developed", "shipped",
    "debugged", "optimized", "refactored", "engineered", "programmed", "designed and built")
# "AI experience consists primarily of recent (<12mo) LangChain-to-OpenAI"
LLM_RECENT_TERMS = ("langchain", "llamaindex", "llama index", "openai api", "prompt engineering",
    "chatgpt", "gpt-4", "gpt-3.5", "rag pipeline")
PRE_LLM_ML_TERMS = ("xgboost", "lightgbm", "random forest", "svm", "logistic regression",
    "gradient boosting", "scikit", "collaborative filtering", "word2vec", "lstm", "cnn",
    "matrix factorization", "feature engineering")
PRODUCTION_TERMS = ("production", "deployed", "shipped", "real users", "at scale", "serving",
    "inference", "latency", "live", "rollout", "monitoring")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _descriptions(candidate: Dict[str, Any]) -> str:
    return " ".join(j.get("description", "") for j in candidate.get("career_history", [])).lower()


def _title(candidate: Dict[str, Any]) -> str:
    return candidate.get("profile", {}).get("current_title", "").lower()


def _all_titles(candidate: Dict[str, Any]) -> List[str]:
    t = [candidate.get("profile", {}).get("current_title", "")]
    t += [j.get("title", "") for j in candidate.get("career_history", [])]
    return [x.lower() for x in t if x]


def evidence_hits(candidate: Dict[str, Any], spec: JDSpec) -> List[str]:
    """All JD evidence terms (core + supporting) appearing in the descriptions."""
    desc = _descriptions(candidate)
    return [t for t in spec.evidence_terms if t in desc]


def core_evidence_hits(candidate: Dict[str, Any], spec: JDSpec) -> List[str]:
    """Domain-specific evidence only — the trustworthy relevance signal."""
    desc = _descriptions(candidate)
    return [t for t in spec.core_evidence_terms if t in desc]


def title_match(candidate: Dict[str, Any], spec: JDSpec) -> bool:
    t = _title(candidate)
    return any(k in t for k in spec.role_titles)


def adjacent_match(candidate: Dict[str, Any], spec: JDSpec) -> bool:
    t = _title(candidate)
    return any(k in t for k in spec.adjacent_titles)


# ---------------------------------------------------------------------------
# honeypot detection (surgical)
# ---------------------------------------------------------------------------
def detect_honeypot(candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    skills = candidate.get("skills", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    total_months = yoe * 12

    # Signal A: >=3 expert skills with 0 months duration (genuinely impossible).
    expert_zero = [s.get("name") for s in skills
                   if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0]
    if len(expert_zero) >= 3:
        reasons.append(f"{len(expert_zero)} expert skills with 0 months used")

    # Signal B: a skill used far longer than the entire career (impossible).
    worst_excess = 0
    worst_skill = None
    for s in skills:
        excess = s.get("duration_months", 0) - total_months
        if excess > worst_excess:
            worst_excess, worst_skill = excess, s.get("name")
    if worst_excess > 48:
        reasons.append(f"skill '{worst_skill}' used {worst_excess} mo beyond {yoe:.0f}y career")

    # Signal C: combination of milder impossibilities.
    if len(expert_zero) >= 2 and worst_excess > 24:
        reasons.append("multiple skill/duration impossibilities")

    is_hp = len(reasons) > 0
    return is_hp, reasons


def honeypot_suspicion(candidate: Dict[str, Any]) -> float:
    """Graded 0-1 suspicion for borderline profiles (used as a soft down-weight)."""
    skills = candidate.get("skills", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    total_months = yoe * 12
    score = 0.0
    expert_zero = sum(1 for s in skills
                      if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    score += min(expert_zero * 0.2, 0.5)
    worst_excess = max((s.get("duration_months", 0) - total_months for s in skills), default=0)
    if worst_excess > 36:
        score += 0.3
    elif worst_excess > 24:
        score += 0.15
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# disqualifiers (only those the JD activates, via spec.disqualifier_concepts)
# ---------------------------------------------------------------------------
def _is_consulting(company: str) -> bool:
    c = company.lower()
    return any(f in c for f in CONSULTING_FIRMS)


def detect_disqualifiers(candidate: Dict[str, Any], spec: JDSpec) -> Tuple[List[str], float]:
    """Return (active disqualifier names, multiplicative penalty in 0..1)."""
    flags: List[str] = []
    penalty = 0.0
    active = set(spec.disqualifier_concepts)
    desc = _descriptions(candidate)
    companies = [j.get("company", "") for j in candidate.get("career_history", [])]
    titles = _all_titles(candidate)

    # keyword-stuffer: non-tech title carrying AI skills but no evidence in descriptions.
    # (The JD's central trap; relevant whenever the role is an AI/ML role.)
    t = _title(candidate)
    if any(tok in t for tok in NON_TECH_TITLE_TOKENS) and not core_evidence_hits(candidate, spec):
        names = [s.get("name", "").lower() for s in candidate.get("skills", [])]
        ai_terms = ["pytorch", "tensorflow", "machine learning", "deep learning", "nlp",
                    "llm", "rag", "embeddings", "transformers", "fine-tuning", "vector"]
        ai_skills = sum(1 for n in names if any(a in n for a in ai_terms))
        if ai_skills >= 5:
            flags.append("keyword_stuffer")
            penalty = max(penalty, 0.97)

    if "only_consulting" in active and companies and all(_is_consulting(c) for c in companies):
        flags.append("only_consulting")
        penalty = max(penalty, 0.6)

    if "title_chaser" in active:
        durations = [j.get("duration_months", 0) for j in candidate.get("career_history", [])]
        if len(durations) >= 3 and (sum(durations) / len(durations)) < 18:
            flags.append("title_chaser")
            penalty = max(penalty, 0.35)

    if "research_only_no_production" in active:
        all_research = bool(titles) and all("research" in t for t in titles)
        if all_research and not any(p in desc for p in PRODUCTION_TERMS):
            flags.append("research_only_no_production")
            penalty = max(penalty, 0.6)

    # CV/speech/robotics primary expertise without NLP/IR exposure (JD explicit).
    # Keyed on the TITLE + absence of retrieval/ranking evidence. Verified against the
    # pool: the 132 CV-titled candidates here all carry retrieval/ranking evidence in
    # their descriptions, so none are flagged (they aren't the "CV-without-IR" type) and
    # are already ranked correctly by evidence/semantic. This fires on genuine CV-only
    # profiles in real data; kept for JD completeness.
    if "wrong_domain_no_nlp" in active:
        cv_title = any(x in t for x in ("computer vision", "cv engineer", "vision engineer",
                                        "speech", "robotics"))
        cv_desc = sum(1 for x in CV_SPEECH_ROBOTICS if x in desc) >= 2
        if (cv_title or cv_desc) and not core_evidence_hits(candidate, spec):
            flags.append("cv_speech_robotics_no_nlp")
            penalty = max(penalty, 0.5)

    # Senior in a leadership/architecture title with no recent hands-on code
    # (JD: "hasn't written production code in the last 18 months"). NOTE: this synthetic
    # pool contains zero leadership titles, so this never fires here; kept for JD
    # completeness / generalization to real data.
    if "no_recent_code" in active and any(l in t for l in LEADERSHIP_TITLES):
        hist = candidate.get("career_history", [])
        recent_desc = (hist[0].get("description", "").lower() if hist else "")
        if not any(v in recent_desc for v in HANDS_ON_VERBS) \
                and not any(p in recent_desc for p in PRODUCTION_TERMS):
            flags.append("stale_leadership_no_hands_on")
            penalty = max(penalty, 0.5)

    # "AI experience" is only recent LLM/LangChain tooling with no pre-LLM ML depth.
    if "only_recent_llm" in active:
        yoe = candidate.get("profile", {}).get("years_of_experience", 0) or 0
        if yoe < 3:
            llm = sum(1 for x in LLM_RECENT_TERMS if x in desc)
            pre = sum(1 for x in PRE_LLM_ML_TERMS if x in desc)
            if llm >= 1 and pre == 0:
                flags.append("only_recent_llm")
                penalty = max(penalty, 0.4)

    return flags, penalty


# ---------------------------------------------------------------------------
# prefilter (Stage-1 recall)
# ---------------------------------------------------------------------------
def passes_prefilter(candidate: Dict[str, Any], spec: JDSpec) -> bool:
    """Keep candidates plausibly relevant to the JD; drop honeypots and pure noise."""
    is_hp, _ = detect_honeypot(candidate)
    if is_hp:
        return False
    if title_match(candidate, spec):
        return True
    # Disguised builder: real domain evidence in descriptions regardless of title.
    if core_evidence_hits(candidate, spec):
        return True
    return False
