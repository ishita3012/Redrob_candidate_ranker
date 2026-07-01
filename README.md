# Intelligent Candidate Ranker — Redrob Hackathon

A JD-agnostic, two-stage **retrieve → rank** system that ranks 100,000 candidates
against a job description by reasoning about *evidence* and *availability*, not keyword
counts. Built so the JD is **data flowing through a general pipeline**, not constants
baked into the code.

> One-line reproduce (CPU-only, no network):
> ```bash
> python src/rank.py --candidates ../candidates.jsonl --jd job_description.md --out submission.csv
> ```
> ~14s on the full 100K pool · validator passes · 0 honeypots in top 100.

---

## The problem (and the traps)

The dataset is built to punish keyword matching. Our own analysis of the 100K pool:

| Finding | Implication for the design |
|---|---|
| Only **0.9%** have a genuine ML title; 99% are decoy roles (HR, sales, ops, …) | The relevant pool is tiny — recall must be precise |
| Skills lists are **uniform noise** (every skill ~12,000 occurrences) | **Never score the skills list** — it's bait for keyword-stuffers |
| **5,425** keyword-stuffers (non-tech title + 5+ AI skills) | Must be gated, not just down-weighted |
| ~**80** honeypots with subtly impossible profiles | Surgical detection; >10% in top 100 = disqualified |
| Real evidence lives in **career-history descriptions**, not skills | Descriptions + title are the trustworthy content signal |

So the right answer is the one the JD describes: *"reason about the gap between what the
JD says and what it means"*, and *"down-weight"* candidates who aren't actually available.

---

## Approach: a two-stage retrieve → rank pipeline

```
OFFLINE (once, network allowed)        ONLINE (rank.py — CPU only, no network, ~14s)
─────────────────────────────          ─────────────────────────────────────────────
precompute.py                          jd_spec.py   parse JD → JDSpec (no hardcoding)
  MiniLM embeddings of evidence text       │
  → artifacts/*.npy                     STAGE 1 — Recall  (gates.py)
                                          drop honeypots (hard) · keep title-match
                                          ∪ core-evidence-in-descriptions  → ~2.6K
                                            │
                                        STAGE 2 — Rerank  (features.py + scoring.py)
                                          fit = Σ(feature × weight)
                                          composite = fit × availability
                                                          × (1 − disqualifier)
                                                          × authenticity
                                          sort → top 100 → reasoning → submission.csv
```

Full design rationale and per-module detail is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

### Key design decisions (each grounded in the data, not taste)

1. **The JD is parsed, not hardcoded.** `jd_spec.py` turns any JD into a structured
   spec (role titles, experience band, locations, must-have/disqualifier *concepts*,
   evidence terms). Swap the JD file → new ranking, **no code change**.
2. **Evidence over skills.** Fit is driven by domain-specific proof in career
   descriptions; the skills list (proven noise) only weakly corroborates.
3. **Behavioral is a multiplier, not a flat term.** The JD says *down-weight*
   unavailable people — so availability modifies fit (≈0.5–1.10×), calibrated to the
   pool's measured percentiles, rather than adding a fixed 25%.
4. **Traps are gated, not nudged.** Honeypots and keyword-stuffers are removed at
   recall, so they cannot reach the top by construction.
5. **Surgical honeypot detection.** Only sharp impossibilities (expert skill with 0
   months used; a skill used longer than the entire career). We explicitly *dropped*
   a grad-year heuristic that false-flagged ~21,000 legitimate candidates.

### Behavioral signals actually change the ranking (real example)

A **Senior NLP Engineer (6.8y)** ranks **#4 on pure fit** — strong retrieval/eval
evidence at a product company. But a **7% recruiter response rate** and **90-day notice**
give a **0.62× availability multiplier**, dropping them **out of the top 100**: perfect
on paper, not actually hireable. Conversely an **86%-response ML Engineer** rises from
fit-#133 to **#66**. This is the differentiator the challenge is fishing for.

---

## Results (verified on the full 100K)

| Metric | Value |
|---|---|
| Ranking-step runtime | ~14s (budget: 5 min) |
| Peak memory | < 2 GB (budget: 16 GB) |
| Honeypots in top 100 | 0 |
| Stage-1 recall pool | 2,658 candidates |
| Reasoning uniqueness | 100/100 distinct |
| Top-100 ML-titled / India-based | 89 / 90 |
| Validator | **Submission is valid.** |

---

## Reproduce

```bash
pip install -r requirements.txt           # only needed for embeddings/sandbox

# Default: TF-IDF semantic backend — standard library only, fully reproducible
python src/rank.py --candidates ../candidates.jsonl --jd job_description.md --out submission.csv

# Validate
python ../validate_submission.py submission.csv
```

**Optional embeddings upgrade** (sentence-transformers; run once, offline/network allowed):
```bash
python src/precompute.py --candidates ../candidates.jsonl --jd job_description.md   # → artifacts/
python src/rank.py --candidates ../candidates.jsonl --jd job_description.md \
                   --out submission.csv --backend stmodel
```
The embeddings are a quality upgrade; the system runs fully on the TF-IDF backend with
no model or network.

---

## Compute constraints

| Constraint | Limit | This system |
|---|---|---|
| Runtime | ≤ 5 min | ~14s |
| Memory | ≤ 16 GB | < 2 GB |
| Compute | CPU only | ✅ |
| Network (rank step) | off | ✅ (embeddings precomputed offline) |

---

## Sandbox / demo

`app.py` is a Streamlit app (upload a candidate sample → ranked CSV + per-candidate
score breakdown). Run locally with `streamlit run app.py`; deploy to Hugging Face
Spaces (see `ARCHITECTURE.md` → *Deploying the sandbox*).

**Live demo:** _<add your Hugging Face Space URL here>_

---

## Repository layout

```
Submission/
├── job_description.md        # JD = pipeline input (parsed, not hardcoded)
├── sample_candidates.json    # small sample for the sandbox
├── app.py                    # Streamlit sandbox
├── requirements.txt
├── ARCHITECTURE.md
└── src/
    ├── jd_spec.py            # JD → structured JDSpec (concept library)
    ├── gates.py             # Stage-1: honeypot detection + prefilter
    ├── features.py          # Stage-2: per-candidate fit features
    ├── scoring.py           # composite = fit × availability × (1−disq) × authenticity
    ├── semantic.py          # evidence-vs-JD similarity (TF-IDF / embeddings)
    ├── reasoning.py         # specific, varied, honest justifications
    ├── ranker.py            # two-stage orchestration
    ├── rank.py              # CLI entry point (single reproduce command)
    └── precompute.py        # offline embedding generation
```

---

## AI tools declaration

Developed with AI assistance (Claude) for design discussion, code, and docs; all
engineering decisions were validated against the actual dataset (see commit history for
the real iteration). No candidate data was sent to any hosted service — ranking is fully
local.
