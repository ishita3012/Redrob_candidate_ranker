"""
Test suite targeting the failure modes that actually disqualify a submission:
honeypots reaching the top, keyword-stuffers surviving, prefilter correctness,
the grad-year false-positive we fixed, tie-break ordering, CSV format, and
reasoning honesty.

    pip install pytest && python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from jd_spec import load_jd                                    # noqa: E402
from gates import (detect_honeypot, passes_prefilter,          # noqa: E402
                   detect_disqualifiers, honeypot_suspicion)
from scoring import compute_relevance, reachability            # noqa: E402
from ranker import rank_candidate_list                         # noqa: E402
from reasoning import generate_reasoning                       # noqa: E402


@pytest.fixture(scope="module")
def spec():
    return load_jd(str(ROOT / "job_description.md"))


def make_candidate(cid="CAND_0000001", title="Machine Learning Engineer", yoe=7.0,
                   descriptions=("Built and shipped a production ranking and retrieval "
                                 "system with embeddings and semantic search.",),
                   skills=None, companies=("Flipkart",), signals=None,
                   education_end=2015, durations=None):
    durations = durations or [36] * len(descriptions)
    hist = []
    for i, d in enumerate(descriptions):
        hist.append({
            "company": companies[i] if i < len(companies) else companies[-1],
            "title": title, "start_date": "2019-01-01", "end_date": None,
            "duration_months": durations[i], "is_current": i == 0,
            "industry": "Internet", "company_size": "1001-5000", "description": d,
        })
    sig = {
        "recruiter_response_rate": 0.8, "last_active_date": "2026-06-10",
        "open_to_work_flag": True, "notice_period_days": 30,
        "interview_completion_rate": 0.9, "willing_to_relocate": True,
        "skill_assessment_scores": {}, "github_activity_score": 40,
        "verified_email": True, "verified_phone": True,
    }
    if signals:
        sig.update(signals)
    return {
        "candidate_id": cid,
        "profile": {"current_title": title, "years_of_experience": yoe,
                    "location": "Pune, Maharashtra", "country": "India",
                    "headline": title, "summary": descriptions[0]},
        "career_history": hist,
        "education": [{"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS",
                       "start_year": education_end - 4, "end_year": education_end,
                       "tier": "tier_1"}],
        "skills": skills or [{"name": "Python", "proficiency": "advanced",
                              "endorsements": 20, "duration_months": 60}],
        "redrob_signals": sig,
    }


# ---- JD parsing ----
def test_jd_parses_expected_fields(spec):
    assert spec.exp_band == (5.0, 9.0)
    assert spec.role_titles, "role titles should be populated for an AI/ML JD"
    assert "only_consulting" in spec.disqualifier_concepts
    assert spec.core_evidence_terms, "core evidence terms should be derived from the JD"


# ---- honeypots ----
def test_honeypot_expert_zero_duration():
    c = make_candidate(skills=[{"name": s, "proficiency": "expert",
                                "endorsements": 0, "duration_months": 0}
                               for s in ("RAG", "FAISS", "PyTorch")])
    assert detect_honeypot(c)[0] is True


def test_honeypot_skill_exceeds_career():
    c = make_candidate(yoe=3.0, skills=[{"name": "RAG", "proficiency": "advanced",
                                         "endorsements": 5, "duration_months": 96}])
    assert detect_honeypot(c)[0] is True


def test_legit_candidate_not_honeypot():
    assert detect_honeypot(make_candidate())[0] is False


def test_grad_year_high_experience_not_flagged():
    # The fixed bug: recent degree + senior experience must NOT be a honeypot.
    c = make_candidate(yoe=8.0, education_end=2024)
    assert detect_honeypot(c)[0] is False
    assert honeypot_suspicion(c) < 0.5


# ---- prefilter / traps ----
def test_prefilter_keeps_ml_with_evidence(spec):
    assert passes_prefilter(make_candidate(), spec) is True


def test_prefilter_drops_noise(spec):
    c = make_candidate(title="HR Manager",
                       descriptions=("Managed recruitment and payroll operations.",))
    assert passes_prefilter(c, spec) is False


def test_keyword_stuffer_flagged(spec):
    ai = ["PyTorch", "TensorFlow", "NLP", "LLM", "RAG", "Embeddings"]
    c = make_candidate(title="Marketing Manager",
                       descriptions=("Ran brand campaigns and social media.",),
                       skills=[{"name": s, "proficiency": "expert", "endorsements": 5,
                                "duration_months": 12} for s in ai])
    flags, penalty = detect_disqualifiers(c, spec)
    assert "keyword_stuffer" in flags and penalty >= 0.9


def test_consulting_only_penalized(spec):
    c = make_candidate(companies=("TCS", "Infosys"),
                       descriptions=("Worked on client ML projects.",
                                     "Delivery for banking client."))
    flags, penalty = detect_disqualifiers(c, spec)
    assert "only_consulting" in flags and penalty > 0


# ---- reachability ----
def test_reachability_bounds(spec):
    m, _ = reachability(make_candidate(), spec)
    assert 0.5 <= m <= 1.10


def test_reachability_penalizes_ghost(spec):
    ghost = make_candidate(signals={"recruiter_response_rate": 0.05,
                                    "last_active_date": "2025-01-01",
                                    "open_to_work_flag": False, "notice_period_days": 150})
    active = make_candidate()
    assert reachability(ghost, spec)[0] < reachability(active, spec)[0]


# ---- relevance ----
def test_evidence_beats_no_evidence(spec):
    strong = make_candidate()
    weak = make_candidate(cid="CAND_0000002",
                          descriptions=("Wrote internal CRUD dashboards.",))
    assert (compute_relevance(strong, spec, 0.7)["relevance"]
            > compute_relevance(weak, spec, 0.7)["relevance"])


# ---- end-to-end format + tie-break ----
def test_output_format_and_monotonic(spec):
    cands = [make_candidate(cid=f"CAND_{i:07d}", yoe=5 + i % 4) for i in range(1, 21)]
    rows, _ = rank_candidate_list(cands, spec, backend="tfidf", top_n=10)
    assert len(rows) == 10
    assert [r["rank"] for r in rows] == list(range(1, 11))
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_tiebreak_by_candidate_id(spec):
    # two identical profiles differing only by id must order by id ascending on ties
    a = make_candidate(cid="CAND_0000009")
    b = make_candidate(cid="CAND_0000008")
    rows, _ = rank_candidate_list([a, b], spec, backend="tfidf", top_n=2)
    if rows[0]["score"] == rows[1]["score"]:
        assert rows[0]["candidate_id"] < rows[1]["candidate_id"]


# ---- reasoning honesty ----
def test_reasoning_uses_real_facts(spec):
    c = make_candidate(title="Senior NLP Engineer", yoe=6.5)
    b = compute_relevance(c, spec, 0.7)
    text = generate_reasoning(c, b, rank=1)
    assert "Senior NLP Engineer" in text and "6.5" in text


def test_reasoning_varies(spec):
    cands = [make_candidate(cid=f"CAND_{i:07d}", title=t, yoe=y)
             for i, (t, y) in enumerate([("ML Engineer", 7), ("Senior NLP Engineer", 8),
                                         ("AI Engineer", 6), ("Data Scientist", 9)], 1)]
    rows, _ = rank_candidate_list(cands, spec, backend="tfidf", top_n=4)
    texts = [r["reasoning"] for r in rows]
    assert len(set(texts)) == len(texts), "reasoning must not be templated/identical"
