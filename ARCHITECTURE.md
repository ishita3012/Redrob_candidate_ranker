# Architecture

A JD-agnostic, two-stage **retrieve → rank** pipeline. This document describes what each
module actually does, why it does it that way (grounded in the dataset), and how the
system generalizes beyond the single JD it's scored on.

---

## 1. Design principles

1. **The JD is data, not code.** Every JD-specific value (titles, experience band,
   locations, must-haves, disqualifiers) is *parsed* from the JD text into a `JDSpec`.
   The ranking logic consumes the spec and never references the JD directly. Test:
   *drop in a different JD and get a sensible ranking without touching the code.*
2. **Recall then precision.** A cheap, high-recall Stage-1 reduces 100K → ~2.6K; an
   expensive, precise Stage-2 reranks only those. This is the retrieval→reranking
   pattern the role itself is about.
3. **Trust evidence, distrust claims.** Career-history *descriptions* and *titles* are
   the content signal. The skills list is empirically noise and is never scored
   directly — embedding or counting it is exactly the trap the dataset sets.
4. **Gates for traps, multipliers for availability, weights for fit.** Different signal
   types enter the score in the form that matches how the JD frames them.

---

## 2. Empirical grounding

Design decisions were calibrated against a full-pool analysis (`analysis` scripts), not
assumed:

| Observation | Decision it drove |
|---|---|
| 0.9% ML-titled; 99% decoy roles | Tight, title/evidence-anchored recall |
| Skills appear ~12,000× each, uniformly | Skills never drive fit; only corroborate |
| Non-ML titles contain **zero** retrieval/ranking evidence in descriptions | "Core evidence" in descriptions is a clean relevance signal |
| Honeypot signals form sharp cliffs (e.g. 99,979 have 0 expert-zero-duration skills, then 21 jump to ≥3) | Detect via sharp impossibilities, high precision |
| A grad-year-vs-experience heuristic flags ~21,700 people | **Rejected** — it's noise, not honeypots |
| response p50=0.44 / p75=0.62 / p90=0.73; inactive median 130d; notice median 90d | Availability thresholds set to real percentiles |

---

## 3. Stage 0 — JD parsing (`jd_spec.py`)

`parse_jd(text) → JDSpec`. Deterministic and local (no network, no LLM).

- A reusable **concept library** maps recruiter-domain language to candidate-detectable
  signals: each concept has `jd_cues` (phrases that activate it from the JD) and
  `evidence` (phrases to look for in candidate descriptions). The **JD text selects
  which concepts apply and with what polarity** (must-have / nice-to-have /
  disqualifier), so a different JD yields a different spec.
- Evidence terms are split into **core** (domain-specific: embeddings, vector DB,
  ranking/recsys, evaluation, LLM) and **supporting** (production, python). Only core
  evidence decides relevance — supporting words like "production"/"python" appear in
  non-ML text and would otherwise leak decoys into recall.
- Also extracts: role + adjacent titles, experience band + ideal sub-band, preferred
  cities + country, notice preference, behavioral priorities, and the JD's
  "ideal-candidate" paragraph (used as the semantic query).

For *this* JD the parse yields exp band (5,9)/ideal (6,8); cities pune/noida/hyderabad/
mumbai/delhi; six must-have concepts; seven disqualifiers — all from the text.

---

## 4. Stage 1 — Recall (`gates.py`)

**Honeypot detection** (`detect_honeypot`) — surgical, high precision. Flags only sharp
impossibilities:
- ≥3 *expert*-proficiency skills with **0 months** used, or
- a skill used **>48 months beyond** the candidate's entire career, or
- a combination of milder versions of the above.

It deliberately does **not** use grad-year-vs-experience (which false-flags ~21K real
candidates). On the full pool this flags ~26 candidates with no false positives observed;
the remaining honeypots also fail on fit, so none reach the top 100.

**Prefilter** (`passes_prefilter`) keeps a candidate iff it is **not** a honeypot **and**
(its current title matches the JD's role/adjacent titles **or** its descriptions contain
core evidence). On this JD: **2,658** survivors — a clean pool of ML/Data/Software/AI
engineers, including "disguised builders" (e.g. Data Engineers who actually built
ranking systems) pulled in by evidence rather than excluded by title.

**Disqualifiers** (`detect_disqualifiers`) activate only those the JDSpec lists:
keyword-stuffer (non-tech title + AI skills + no real evidence), only-consulting,
title-chaser, research-only-no-production. Each contributes a multiplicative penalty.

---

## 5. Stage 2 — Rerank (`features.py`, `scoring.py`)

**Fit features** (each 0–1, with evidence strings for reasoning):

| Feature | Weight | What it measures |
|---|---|---|
| `evidence` | 0.28 | Count/depth of core domain evidence in descriptions |
| `title` | 0.20 | Role/adjacent title match, seniority, prior ML roles |
| `semantic` | 0.17 | Similarity of evidence text to the JD ideal-candidate text |
| `product` | 0.11 | Fraction of career at product (non-consulting) companies |
| `experience` | 0.09 | Smooth fit to the JD band (soft, decays outside) |
| `stability` | 0.07 | Tenure pattern — demotes title-chasers (JD rejects ~1.5y switching) |
| `location` | 0.05 | Preferred city / in-country / relocatable / outside |
| `skill_corroboration` | 0.03 | Platform-verified assessment scores only |

Weights reflect the empirical reality that evidence + title carry the signal and skills
are noise.

**Score** (`compute_relevance`) — a two-factor model from the JD's own "great fit but
not actually available" framing:
```
score        = relevance × reachability
relevance    = trust × Σ(feature × weight) × (1 − red_flag_penalty)
reachability = behavioral availability, bounded ≈[0.5, 1.10]
```
- **relevance**: how well the candidate fits the JD, discounted by `trust`
  (`1 − 0.5 × honeypot_suspicion`, for borderline profiles) and by the JD's explicit
  red flags (consulting-only, title-chasing, research-only — which also surface directly
  in the `product`, `stability`, and `evidence` features).
- **reachability** (`reachability`): response rate, recency, open-to-work, notice,
  interview-completion — thresholds set to the pool's measured percentiles. A modifier,
  per the JD, not an additive term. Traps (honeypots, keyword-stuffers) are removed by
  the Stage-1 gates, not modelled here.

---

## 6. Semantic similarity (`semantic.py`)

Embeds **evidence text** (career descriptions + summary + titles), *not* the skills list,
against the JD's ideal-candidate paragraph. Two CPU-only, network-free backends:
- **`tfidf`** (default): a local TF-IDF cosine over the recalled pool. Zero
  dependencies, fully reproducible, no model download.
- **`stmodel`** (optional): precomputed `all-MiniLM-L6-v2` embeddings from
  `precompute.py` (offline). Catches oblique phrasing the lexical backend misses, and
  demonstrates the embeddings-based retrieval the JD asks for. Online, only a dot
  product runs.

---

## 7. Reasoning (`reasoning.py`)

Stage-4 review samples 10 rows and checks specificity, JD connection, honest concerns,
no hallucination, variation, and rank-consistency. Reasoning is assembled **only** from
the candidate's actual fields and the computed breakdown — never invented — leads with
concrete identity + strongest evidence, includes real behavioral numbers, and **states
genuine concerns** (out-of-band experience, geography, long notice, low response,
adjacent title, consistency flags). On the full run all 100 reasonings are distinct.

---

## 8. Generalization (and its honest boundary)

The **pipeline** is JD-agnostic by construction: titles, bands, locations, concepts,
evidence terms, and disqualifiers all come from the parsed `JDSpec`. The **vocabulary**
it draws on is domain-scoped, and we're explicit about that rather than claiming "works
on any JD":

- **Fully general:** `semantic` (embeds any JD text) and `experience` (parses any range).
- **AI/ML-scoped, with graceful fallback:** `title` (role vocab) and `evidence` (concept
  library) are AI/ML. When the library doesn't recognize a JD's domain, `parse_jd`
  derives fallback role-title keys from the JD's title line and fallback evidence
  keywords from the JD text (`_fallback_role_titles`, `_fallback_evidence_terms`), so a
  novel-domain JD degrades gracefully instead of scoring zero. **These fire only when the
  primary path is empty**, so a covered JD is provably unaffected (byte-identical output
  verified on this JD).
- **Geo/domain-scoped:** `location` (Indian-city gazetteer), `product` (India consulting
  list), `skill_corroboration` (ML assessment keywords). Out-of-scope inputs go neutral.

So: swapping in **another AI/ML JD** needs no code change; supporting a **new domain** is
a matter of **extending the concept library and gazetteers — data, not logic**.

---

## 9. Compute & reproducibility

- Ranking step: **CPU-only, no network**, ~16s on 100K, < 2 GB RAM.
- Single reproduce command: `python src/rank.py --candidates <file> --jd <file> --out <file>`.
- Embedding precomputation is the only step that may use the network, and it is offline
  and optional; the default TF-IDF backend needs nothing beyond the standard library.

---

## 10. Deploying the sandbox (Hugging Face Spaces)

Create a Streamlit Space and push the contents of `Submission/`. The Space's `README.md`
needs this front-matter header:

```yaml
---
title: Intelligent Candidate Ranker
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---
```

The Space uses the TF-IDF backend (no model download), accepts a candidate sample
(bundled or uploaded), and returns a ranked CSV with per-candidate score breakdowns —
satisfying the small-sample reproducibility the sandbox requirement asks for.

---

## 11. Design tradeoffs (for defend-your-work)

- **Why not a learning-to-rank model?** No labels exist; an LTR model would fit our own
  heuristic back to itself (circular) and overfit our assumptions. A transparent,
  JD-grounded scoring function is more honest and more defensible.
- **Why gate honeypots instead of relying on the reranker?** A reranker that reads
  embeddings can still rank a honeypot with high similarity; gating removes them by
  construction and protects the >10% disqualification threshold.
- **Why embed evidence, not the whole profile?** Embedding the skills list rewards
  keyword-stuffers — the central trap. Evidence text is the trustworthy signal.
- **Why behavioral as a multiplier?** The JD frames availability as a down-weight on
  otherwise-good candidates, which is multiplicative, not a separate additive axis.
