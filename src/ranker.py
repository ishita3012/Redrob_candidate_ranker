"""
Two-stage ranking orchestration:  recall (gates) -> rerank (scoring) -> top 100.

Tie-break: scores are rounded to the output precision BEFORE sorting, and the sort
key is (-score, candidate_id). This guarantees the submission spec's rule that equal
scores are ordered by candidate_id ascending, and that score is non-increasing.
"""

from __future__ import annotations
import csv
import json
import time
from typing import Dict, Any, List, Tuple

from jd_spec import load_jd, JDSpec
from gates import passes_prefilter
from semantic import compute_semantic_scores
from scoring import compute_relevance
from reasoning import generate_reasoning

SCORE_DECIMALS = 6


def stream_candidates(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def rank(candidates_path: str, jd_path: str, out_path: str,
         backend: str = "tfidf", top_n: int = 100, verbose: bool = True
         ) -> Dict[str, Any]:
    t0 = time.time()
    spec = load_jd(jd_path)
    if verbose:
        print(spec.summary())

    # ---- Stage 1: recall ----
    survivors = [c for c in stream_candidates(candidates_path) if passes_prefilter(c, spec)]
    if verbose:
        print(f"\n[recall] {len(survivors)} candidates passed Stage-1 gates "
              f"({time.time()-t0:.1f}s)")

    # ---- semantic scores over the recalled pool ----
    sem = compute_semantic_scores(survivors, spec, backend=backend)

    # ---- Stage 2: rerank ----
    scored: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    for c in survivors:
        b = compute_relevance(c, spec, sem.get(c["candidate_id"], 0.5))
        score = round(b["composite"], SCORE_DECIMALS)
        scored.append((c, b, score))

    scored.sort(key=lambda x: (-x[2], x[0]["candidate_id"]))
    top = scored[:top_n]

    rows = []
    for rk, (c, b, score) in enumerate(top, start=1):
        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rk,
            "score": score,
            "reasoning": generate_reasoning(c, b, rk, top_n),
        })

    _save(rows, out_path)
    if verbose:
        print(f"[done] wrote {len(rows)} rows to {out_path} in {time.time()-t0:.1f}s")

    return {"spec": spec, "survivors": survivors, "top": top, "rows": rows,
            "elapsed": time.time() - t0}


def _save(rows: List[Dict[str, Any]], out_path: str):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in rows:
            w.writerow([r["candidate_id"], r["rank"],
                        f"{r['score']:.{SCORE_DECIMALS}f}",
                        r["reasoning"].replace("\n", " ").strip()])
