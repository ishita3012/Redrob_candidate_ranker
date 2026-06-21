# Intelligent Candidate Ranker

## Redrob Hackathon - Data & AI Challenge

An evidence-based candidate ranking system that revolutionizes hiring by going far beyond keyword matching.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ranker
cd src
python main.py --candidates ../../candidates.jsonl --output ../submission.csv

# 3. Validate submission
cd ..
python ../validate_submission.py submission.csv

# 4. (Optional) Run the demo app
streamlit run app.py
```

**That's it!** The ranker processes 100K candidates in ~40 seconds.

---

## The Problem We're Solving

Traditional ATS systems fail because they use **keyword matching**, which:

1. **Rewards keyword stuffing** — A Marketing Manager with "PyTorch, TensorFlow, NLP" in their skills beats actual ML engineers
2. **Ignores behavioral signals** — A perfect-on-paper candidate who hasn't logged in for 6 months isn't actually available
3. **Can't detect impossible profiles** — Honeypots with fabricated experience slip through
4. **Misses hidden gems** — Backend Engineers who built recommendation systems aren't found because they don't have "ML" in their title

---

## Our Solution: Evidence-Based Ranking

We score candidates across **8 dimensions**, each addressing a failure mode:

### Scoring Dimensions

| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| **Semantic** | 15% | Embedding similarity to JD |
| **Role Fit** | 15% | Title alignment, experience range |
| **Skill Match** | 15% | Skills with trust verification |
| **ML Evidence** | 15% | Proof of work in career descriptions |
| **Behavioral** | 25% | Response rate, activity, availability |
| **Trajectory** | 10% | Career progression, stability |
| **Education** | 3% | Institution tier, field relevance |
| **GitHub** | 2% | Coding activity |

### Key Innovations

1. **Skill Trust Scoring**
   - We don't just check if a skill is listed
   - We verify through: duration, endorsements, assessments, career evidence
   - A skill with 5 years + 30 endorsements + 85% assessment >> skill with 0 evidence

2. **Behavioral Signals (25% of score)**
   - `recruiter_response_rate`: Will they reply?
   - `last_active_date`: Are they looking?
   - `notice_period_days`: Can they join quickly?
   - `interview_completion_rate`: Will they show up?

3. **Honeypot Detection**
   - Expert skills with 0 duration
   - Skill duration > total experience
   - Impossible career timelines

4. **Disqualifier Detection**
   - Keyword stuffers (irrelevant title + AI skills)
   - Only consulting experience
   - Title chasers (avg tenure < 18 months)
   - Research-only (no production)

---

## Architecture

```
Submission/
├── app.py                 # Streamlit demo (sandbox)
├── requirements.txt       # Dependencies
├── README.md             # This file
└── src/
    ├── config.py         # JD requirements, weights, thresholds
    ├── data_loader.py    # Load and parse candidates
    ├── honeypot_detector.py  # Detect impossible profiles
    ├── disqualifiers.py  # JD-specific red flags
    ├── scorers.py        # Multi-dimensional scoring
    ├── semantic.py       # Embedding-based matching (optional)
    ├── ranker.py         # Main ranking logic
    └── main.py           # CLI entry point
```

---

## Results

On the full 100K candidate dataset:

| Metric | Value |
|--------|-------|
| Processing time | ~40 seconds |
| Honeypots in top 100 | 0 |
| Keyword stuffers in top 100 | 0 |
| ML-titled candidates in top 100 | 100% |
| Average response rate | 69% |
| Average experience | 6.5 years |
| Validation | PASSED |

---

## How Each Component Works

### 1. Role Fit (`score_role_fit`)

```python
# Checks:
# - Is current title ML-relevant? (+0.4)
# - ML roles in career history? (+0.1 each, max 0.3)
# - Experience in 5-9 year range? (+0.2)
# - In preferred locations? (+0.1)
```

### 2. Skill Trust (`compute_skill_trust`)

```python
# For each skill:
trust = 0.1  # Base: listed it
trust += min(duration_months / 48, 0.2)      # Duration
trust += min(endorsements / 30, 0.15)         # Endorsements
trust += (assessment_score / 100) * 0.3       # Platform verification
trust += 0.1 if in_job_descriptions else 0    # Evidence of use
```

### 3. Honeypot Detection

```python
# Detected if:
# - 3+ expert skills with 0 months duration
# - Skill duration > total experience by 24+ months
# - Experience > years since graduation
```

### 4. Behavioral Scoring

```python
# Response rate: 0.7+ is good, <0.2 is red flag
# Last active: <14 days is good, >180 days is penalty
# Notice period: <30 days preferred, <60 okay
# Open to work: bonus if true
```

---

## Reproduction

### Step 1: Clone and Install

```bash
git clone <your-repo>
cd Submission
pip install -r requirements.txt
```

### Step 2: Run Ranking

```bash
cd src
python main.py --candidates ../../candidates.jsonl --output ../submission.csv
```

### Step 3: Validate

```bash
cd ..
python ../validate_submission.py submission.csv
```

Expected output: `Submission is valid.`

---

## Demo / Sandbox

We provide a Streamlit app for interactive exploration:

```bash
streamlit run app.py
```

Features:
- Upload JSON/JSONL candidate files
- See rankings with explanations
- Explore individual candidate scores
- Download results as CSV

**Live demo**: [Link to HuggingFace Space or Streamlit Cloud]

---

## Semantic Embeddings (Optional)

For even better matching, you can enable semantic embeddings:

```bash
# 1. Install dependencies
pip install sentence-transformers numpy

# 2. Pre-compute embeddings (one-time, ~5-10 min)
cd src
python semantic.py --precompute --candidates ../../candidates.jsonl

# 3. Run with semantic matching
python main.py --candidates ../../candidates.jsonl --output ../submission.csv --semantic
```

This adds embedding-based similarity as 15% of the total score.

---

## Compute Constraints

Our system meets all hackathon requirements:

| Constraint | Requirement | Our System |
|------------|-------------|------------|
| Runtime | < 5 minutes | ~40 seconds |
| Memory | < 16 GB | ~2 GB peak |
| Compute | CPU only | ✅ No GPU |
| Network | Offline | ✅ No API calls |

---

## Why This Approach Works

1. **Evidence over claims**: We verify skills through multiple signals, not just listings
2. **Behavioral signals matter**: 25% weight ensures we find *available* candidates
3. **Disqualifiers first**: We filter traps before scoring
4. **Honeypot resistant**: We catch impossible profiles
5. **Explainable**: Every ranking has human-readable reasoning
6. **Fast**: 40 seconds for 100K candidates

---

## Files

| File | Purpose |
|------|---------|
| `submission.csv` | Final ranking output |
| `app.py` | Streamlit demo |
| `src/main.py` | CLI entry point |
| `src/config.py` | All configuration |
| `src/ranker.py` | Main ranking logic |
| `src/scorers.py` | Scoring functions |
| `src/honeypot_detector.py` | Honeypot detection |
| `src/disqualifiers.py` | Red flag detection |
| `src/semantic.py` | Embedding utilities |

---

## AI Tools Declaration

- **Claude**: Architecture design, code review
- **GitHub Copilot**: Autocomplete

No candidate data was sent to any LLM. All ranking is done locally.

---

## Team

*[Fill in your team details in submission_metadata.yaml]*

---

## License

MIT License
