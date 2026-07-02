#!/usr/bin/env python3
"""
Single-command entry point for Stage-3 reproduction.

    python rank.py --candidates ../candidates.jsonl --jd ../job_description.md \
                   --out ../submission.csv

Backends:
    --backend tfidf    (default) local TF-IDF semantic similarity, no model/network
    --backend stmodel  use precomputed sentence-transformers embeddings in artifacts/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ranker import rank


def main():
    ap = argparse.ArgumentParser(description="Intelligent candidate ranker (JD-agnostic)")
    ap.add_argument("--candidates", "-c", required=True, help="path to candidates.jsonl")
    ap.add_argument("--jd", "-j", default="../job_description.md", help="path to JD text")
    ap.add_argument("--out", "-o", default="../submission.csv", help="output CSV path")
    ap.add_argument("--backend", default="auto", choices=["auto", "tfidf", "stmodel"])
    ap.add_argument("--top-n", type=int, default=100)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rank(args.candidates, args.jd, args.out,
         backend=args.backend, top_n=args.top_n, verbose=not args.quiet)


if __name__ == "__main__":
    main()
