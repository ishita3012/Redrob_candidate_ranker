"""
Minimal Streamlit sandbox for the JD-agnostic candidate ranker (Redrob hackathon).

This is the required reproducibility demo, not a product UI: load a small candidate
sample, run the full two-stage pipeline, show and download the ranked CSV. Uses the
TF-IDF backend so the Space needs no model download or network.

    streamlit run app.py
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
st.title("Intelligent Candidate Ranker")
st.caption("JD-agnostic, two-stage retrieve→rank. The JD is parsed from "
           "`job_description.md` — swap the JD, no code changes.")

top_n = st.sidebar.slider("Top-N to return", 5, 100, 25)
uploaded = st.sidebar.file_uploader("Candidates (.json / .jsonl, ≤100)",
                                    type=["json", "jsonl"])

if uploaded is not None:
    raw = uploaded.read().decode("utf-8")
    try:
        candidates = json.loads(raw)
        candidates = candidates if isinstance(candidates, list) else [candidates]
    except json.JSONDecodeError:
        candidates = [json.loads(x) for x in raw.splitlines() if x.strip()]
else:
    candidates = json.load(open(ROOT / "sample_candidates.json"))

spec = load_jd(str(ROOT / "job_description.md"))
n_survive = sum(1 for c in candidates if passes_prefilter(c, spec))
n_hp = sum(1 for c in candidates if detect_honeypot(c)[0])
st.caption(f"{len(candidates)} candidates · {n_survive} passed Stage-1 recall · "
           f"{n_hp} honeypots filtered")

rows, _ = rank_candidate_list(candidates, spec, backend="tfidf", top_n=top_n)
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button("Download submission.csv", df.to_csv(index=False).encode("utf-8"),
                   file_name="submission.csv", mime="text/csv")
