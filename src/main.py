#!/usr/bin/env python3
"""
Intelligent Candidate Ranker - Main Entry Point

Usage:
    # Basic ranking (without semantic embeddings)
    python main.py --candidates ../candidates.jsonl --output ../submission.csv

    # With semantic embeddings (requires pre-computation)
    python main.py --candidates ../candidates.jsonl --output ../submission.csv --semantic

    # Pre-compute embeddings first (one-time, can take several minutes)
    python semantic.py --precompute --candidates ../candidates.jsonl

This script:
1. Loads all candidates from the JSONL file
2. Optionally loads pre-computed semantic embeddings
3. Scores each candidate across multiple dimensions
4. Detects honeypots and applies disqualifiers
5. Generates rankings with reasoning
6. Outputs a submission CSV
"""

import argparse
import time
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_candidates
from ranker import rank_candidates, save_submission, print_ranking_summary, validate_ranking


def main():
    parser = argparse.ArgumentParser(
        description='Intelligent Candidate Ranker for Redrob Hackathon'
    )
    parser.add_argument(
        '--candidates', '-c',
        type=str,
        required=True,
        help='Path to candidates.jsonl file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='submission.csv',
        help='Output path for submission CSV (default: submission.csv)'
    )
    parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=100,
        help='Number of candidates to rank (default: 100)'
    )
    parser.add_argument(
        '--semantic',
        action='store_true',
        help='Use semantic embeddings (requires pre-computation)'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        default='.',
        help='Directory containing pre-computed embeddings'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress information'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("INTELLIGENT CANDIDATE RANKER")
    print("Redrob Hackathon - Data & AI Challenge")
    print("=" * 60)

    # Load candidates
    print(f"\n[1/4] Loading candidates from {args.candidates}...")
    start_time = time.time()
    candidates = load_candidates(args.candidates)
    load_time = time.time() - start_time
    print(f"      Loaded {len(candidates)} candidates in {load_time:.2f}s")

    # Load semantic scores if requested
    semantic_scores = {}
    if args.semantic:
        print(f"\n[2/4] Loading semantic embeddings from {args.embeddings_dir}...")
        try:
            from semantic import compute_semantic_scores
            semantic_start = time.time()
            semantic_scores = compute_semantic_scores(candidates, args.embeddings_dir)
            semantic_time = time.time() - semantic_start
            print(f"      Loaded semantic scores in {semantic_time:.2f}s")
        except Exception as e:
            print(f"      WARNING: Could not load semantic scores: {e}")
            print("      Continuing without semantic scoring...")
    else:
        print("\n[2/4] Skipping semantic embeddings (use --semantic to enable)")

    # Rank candidates
    print(f"\n[3/4] Ranking candidates...")
    rank_start = time.time()
    ranked = rank_candidates(
        candidates,
        semantic_scores=semantic_scores,
        top_n=args.top_n,
        verbose=args.verbose
    )
    rank_time = time.time() - rank_start
    print(f"      Ranking completed in {rank_time:.2f}s")

    # Validate
    errors = validate_ranking(ranked)
    if errors:
        print("\n*** VALIDATION ERRORS ***")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Save submission
    print(f"\n[4/4] Saving submission to {args.output}...")
    save_submission(ranked, args.output)
    print("      Submission saved successfully!")

    # Print summary
    print_ranking_summary(ranked)

    total_time = load_time + rank_time
    print(f"\nTotal processing time: {total_time:.2f}s")
    print("\n" + "=" * 60)
    print("Done! Validate your submission:")
    print(f"  python validate_submission.py {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
