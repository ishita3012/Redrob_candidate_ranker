#!/usr/bin/env python3
"""
OFFLINE precomputation of candidate + JD embeddings (run once; network allowed here).

Produces the artifacts/ used by the 'stmodel' semantic backend at rank time, so the
ranking step itself stays CPU-only and network-free:

    artifacts/candidate_embeddings.npy   (N x 384, float32, L2-normalized)
    artifacts/candidate_ids.json         (aligned candidate_id list)
    artifacts/jd_vector.npy              (384, the JD ideal-candidate embedding)

Usage:
    python precompute.py --candidates ../../candidates.jsonl --jd ../job_description.md

This may exceed the 5-minute online budget (that is fine — it is offline). The online
rank.py --backend stmodel then only does a dot product per candidate.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jd_spec import load_jd
from semantic import candidate_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", "-c", required=True)
    ap.add_argument("--jd", "-j", default="../job_description.md")
    ap.add_argument("--out", "-o", default="artifacts")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    import numpy as np
    from sentence_transformers import SentenceTransformer

    os.makedirs(args.out, exist_ok=True)
    spec = load_jd(args.jd)

    print("Loading candidates...")
    ids, texts = [], []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                ids.append(c["candidate_id"])
                texts.append(candidate_text(c))
    print(f"{len(ids)} candidates. Loading model '{args.model}'...")

    model = SentenceTransformer(args.model)
    t0 = time.time()
    emb = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    np.save(os.path.join(args.out, "candidate_embeddings.npy"), emb.astype("float32"))
    with open(os.path.join(args.out, "candidate_ids.json"), "w") as f:
        json.dump(ids, f)

    jd_vec = model.encode([spec.ideal_text], normalize_embeddings=True,
                          convert_to_numpy=True)[0]
    np.save(os.path.join(args.out, "jd_vector.npy"), jd_vec.astype("float32"))

    print(f"Saved artifacts to {args.out}/ in {time.time()-t0:.1f}s; "
          f"embeddings shape={emb.shape}")


if __name__ == "__main__":
    main()
