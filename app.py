"""
Intelligent Candidate Ranker — simple web app.

Upload a candidate file (.json / .jsonl) and get the top matches back, ranked by an
evidence-based scoring model with behavioral availability weighting. Runs the full
ranking pipeline locally (CPU only, no network).

    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from jd_spec import load_jd                  # noqa: E402
from ranker import rank_candidate_list       # noqa: E402
from scoring import FIT_WEIGHTS              # noqa: E402

st.set_page_config(page_title="Intelligent Candidate Ranker", page_icon="🎯", layout="wide")
st.title("🎯 Intelligent Candidate Ranker")
st.write("Upload a candidate pool and get the strongest matches — ranked on proven "
         "experience and real availability, not keyword counts.")

# --- Scoring model (weighting only; no dataset/role specifics) ---
with st.sidebar:
    st.header("Scoring model")
    st.caption("Final score = **relevance × reachability**")
    st.markdown("**Relevance** — weighted fit signals:")
    st.dataframe(
        pd.DataFrame(
            [{"signal": k, "weight": v} for k, v in FIT_WEIGHTS.items()]
        ).sort_values("weight", ascending=False),
        hide_index=True, use_container_width=True,
    )
    st.markdown("**Reachability** — behavioral availability multiplier "
                "(response rate, recency, notice period, open-to-work).")
    top_n = st.slider("How many top candidates to return", 5, 100, 20)

# --- Single upload ---
uploaded = st.file_uploader("Upload candidates (.json or .jsonl)", type=["json", "jsonl"])

if uploaded is None:
    st.info("⬆️  Upload a candidate file to rank. Each record should include the "
            "candidate's profile, career history, skills, and engagement signals.")
    st.stop()

raw = uploaded.read().decode("utf-8")
try:
    candidates = json.loads(raw)
    candidates = candidates if isinstance(candidates, list) else [candidates]
except json.JSONDecodeError:
    candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]

spec = load_jd(str(ROOT / "job_description.md"))
rows, top = rank_candidate_list(candidates, spec, backend="tfidf", top_n=top_n)

# --- Enriched results for display ---
by_id = {c.get("candidate_id"): c for c in candidates}
display = []
for r in rows:
    c = by_id.get(r["candidate_id"], {})
    p = c.get("profile", {})
    s = c.get("redrob_signals", {})
    display.append({
        "Rank": r["rank"],
        "Candidate": r["candidate_id"],
        "Title": p.get("current_title", ""),
        "Experience": f"{p.get('years_of_experience', '')}y",
        "Location": p.get("location", ""),
        "Response rate": f"{(s.get('recruiter_response_rate') or 0):.0%}",
        "Notice": f"{s.get('notice_period_days', '')}d",
        "Score": r["score"],
        "Why": r["reasoning"],
    })

st.success(f"Ranked {len(candidates)} candidates → showing top {len(rows)}.")
st.dataframe(pd.DataFrame(display), hide_index=True, use_container_width=True)
st.download_button(
    "⬇️ Download ranking (CSV)",
    pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
    file_name="ranking.csv", mime="text/csv",
)
