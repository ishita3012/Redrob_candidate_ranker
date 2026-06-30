"""
Streamlit sandbox for the JD-agnostic candidate ranker (Redrob hackathon).

Demonstrates the full pipeline end-to-end on a small candidate sample:
  parse JD -> Stage-1 recall (gates) -> Stage-2 rerank (fit x availability x ...) -> top-N.

Uses the TF-IDF semantic backend so the Space needs no model download or network.
Run locally:   streamlit run app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from jd_spec import load_jd                          # noqa: E402
from ranker import rank_candidate_list               # noqa: E402
from gates import passes_prefilter, detect_honeypot  # noqa: E402

st.set_page_config(page_title="Intelligent Candidate Ranker", layout="wide")
st.title("Intelligent Candidate Ranker — Redrob Hackathon")
st.caption("JD-agnostic, two-stage retrieve→rank. The JD is parsed into a spec the "
           "ranker consumes — swap the JD, no code changes.")


@st.cache_data
def get_spec_summary():
    return load_jd(str(ROOT / "job_description.md")).summary()


with st.expander("Parsed JD spec (derived from job_description.md, not hardcoded)"):
    st.code(get_spec_summary(), language="text")

st.sidebar.header("Input")
top_n = st.sidebar.slider("Top-N to return", 5, 100, 25)
uploaded = st.sidebar.file_uploader("Upload candidates (.json / .jsonl, ≤100)",
                                    type=["json", "jsonl"])

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    try:
        candidates = json.loads(raw)
        if isinstance(candidates, dict):
            candidates = [candidates]
    except json.JSONDecodeError:
        candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
    st.sidebar.success(f"Loaded {len(candidates)} uploaded candidates")
else:
    candidates = json.load(open(ROOT / "sample_candidates.json"))
    st.sidebar.info(f"Using bundled sample ({len(candidates)} candidates)")

spec = load_jd(str(ROOT / "job_description.md"))

n_hp = sum(1 for c in candidates if detect_honeypot(c)[0])
n_survive = sum(1 for c in candidates if passes_prefilter(c, spec))
col1, col2, col3 = st.columns(3)
col1.metric("Candidates", len(candidates))
col2.metric("Passed Stage-1 recall", n_survive)
col3.metric("Honeypots filtered", n_hp)

rows, top = rank_candidate_list(candidates, spec, backend="tfidf", top_n=top_n)
df = pd.DataFrame(rows)
st.subheader(f"Top {len(rows)} ranking")
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button("Download submission.csv", df.to_csv(index=False).encode("utf-8"),
                   file_name="submission.csv", mime="text/csv")

st.subheader("Score breakdown")
if rows:
    pick = st.selectbox("Inspect a ranked candidate",
                        [f"#{r['rank']} · {r['candidate_id']}" for r in rows])
    idx = int(pick.split("·")[0].strip().lstrip("#")) - 1
    cand, b, score = top[idx]
    p = cand["profile"]
    st.write(f"**{p.get('current_title')}** · {p.get('years_of_experience')}y · "
             f"{p.get('location')}")
    st.json({
        "composite": round(b["composite"], 4),
        "fit": round(b["fit"], 4),
        "availability_multiplier": round(b["availability"], 4),
        "authenticity_multiplier": round(b["authenticity"], 4),
        "disqualifier_penalty": round(b["disqualifier_penalty"], 4),
        **{f"feature:{k}": round(v, 3) for k, v in b["feature_values"].items()},
    })
