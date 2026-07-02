"""
Semantic similarity between a candidate's EVIDENCE text and the JD's ideal-candidate
text. Embedding the evidence (career descriptions + summary), NOT the skills list, is
deliberate: skills lists are noise and embedding them rewards keyword-stuffers.

Two backends, both CPU-only and network-free at rank time:
  * tfidf  (default): a local, dependency-light TF-IDF cosine. Fully reproducible,
    no model download. Computed over the recalled pool, so it is fast and the IDF
    is specific to the candidates being compared.
  * stmodel (optional upgrade): precomputed sentence-transformers embeddings
    (all-MiniLM-L6-v2) loaded from artifacts/. Catches oblique phrasing the lexical
    backend misses. Precompute happens OFFLINE (see precompute.py); only a dot-product
    runs online.

The dense backend doubles as the "dense recall arm" — semantic_scores over the pool
let the ranker pull in disguised builders whose wording isn't in the keyword lexicon.
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import Dict, Any, List

from jd_spec import JDSpec

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\+\#\.]{1,}")
_STOP = set("the a an and or of to for in on at with by is are was were be been being "
            "we our you your i this that these those as it its from into over under "
            "their them they he she his her not no but if then so than also can will "
            "have has had do does did using used use work worked working team teams".split())


def candidate_text(candidate: Dict[str, Any]) -> str:
    p = candidate.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for j in candidate.get("career_history", []):
        parts.append(j.get("title", ""))
        parts.append(j.get("description", ""))
    return " ".join(filter(None, parts)).lower()


def _tok(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


# ---------------------------------------------------------------------------
# TF-IDF backend (default)
# ---------------------------------------------------------------------------
def _tfidf_scores(candidates: List[Dict[str, Any]], jd_text: str) -> Dict[str, float]:
    docs = {c["candidate_id"]: _tok(candidate_text(c)) for c in candidates}
    jd_tokens = _tok(jd_text)

    # document frequency across pool + JD
    df = Counter()
    for toks in docs.values():
        df.update(set(toks))
    df.update(set(jd_tokens))
    N = len(docs) + 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}

    def vec(tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        if not tf:
            return {}
        maxtf = max(tf.values())
        return {t: (0.5 + 0.5 * tf[t] / maxtf) * idf.get(t, 0.0) for t in tf}

    def cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    jd_vec = vec(jd_tokens)
    raw = {cid: cos(vec(toks), jd_vec) for cid, toks in docs.items()}
    # rescale to spread the signal across 0..1 (cosines here are typically small)
    if raw:
        hi = max(raw.values()) or 1.0
        return {cid: (v / hi) for cid, v in raw.items()}
    return raw


# ---------------------------------------------------------------------------
# sentence-transformers backend (optional, precomputed)
# ---------------------------------------------------------------------------
def _stmodel_scores(candidates: List[Dict[str, Any]], spec: JDSpec,
                    artifacts_dir: str) -> Dict[str, float]:
    import numpy as np
    import json
    import os
    emb_path = os.path.join(artifacts_dir, "candidate_embeddings.npy")
    ids_path = os.path.join(artifacts_dir, "candidate_ids.json")
    if not (os.path.exists(emb_path) and os.path.exists(ids_path)):
        raise FileNotFoundError("precomputed embeddings not found")
    emb = np.load(emb_path)
    ids = json.load(open(ids_path))
    id_to_row = {cid: i for i, cid in enumerate(ids)}
    # JD vector: encode once (offline allowed); cached alongside artifacts if present
    jd_vec_path = os.path.join(artifacts_dir, "jd_vector.npy")
    jd_vec = np.load(jd_vec_path)
    jd_vec = jd_vec / (np.linalg.norm(jd_vec) + 1e-9)
    out = {}
    for c in candidates:
        r = id_to_row.get(c["candidate_id"])
        if r is None:
            out[c["candidate_id"]] = 0.5
            continue
        v = emb[r]
        v = v / (np.linalg.norm(v) + 1e-9)
        out[c["candidate_id"]] = float((np.dot(v, jd_vec) + 1) / 2)  # -1..1 -> 0..1
    return out


def compute_semantic_scores(candidates: List[Dict[str, Any]], spec: JDSpec,
                            backend: str = "auto", artifacts_dir: str = "artifacts"
                            ) -> Dict[str, float]:
    """Return {candidate_id: semantic_score in 0..1} over the given (recalled) pool.

    backend="auto" (default) uses precomputed embeddings when artifacts exist, else the
    dependency-free TF-IDF backend — so a repo with artifacts ranks with embeddings and
    one without still reproduces deterministically.
    """
    import os
    if backend == "auto":
        backend = ("stmodel"
                   if os.path.exists(os.path.join(artifacts_dir, "candidate_embeddings.npy"))
                   else "tfidf")
    if backend == "stmodel":
        try:
            return _stmodel_scores(candidates, spec, artifacts_dir)
        except Exception as e:
            print(f"[semantic] stmodel backend unavailable ({e}); falling back to tfidf")
    return _tfidf_scores(candidates, spec.ideal_text)
