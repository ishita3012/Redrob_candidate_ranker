"""
Main ranking module.

Combines all scoring components to produce final rankings with reasoning.
"""

import csv
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from config import SCORING_WEIGHTS
from honeypot_detector import detect_honeypot, get_honeypot_score
from disqualifiers import detect_disqualifiers
from scorers import compute_final_score, compute_all_scores


@dataclass
class RankedCandidate:
    """Represents a ranked candidate with score and reasoning."""
    candidate_id: str
    rank: int
    score: float
    reasoning: str
    is_honeypot: bool
    disqualifiers: List[str]
    score_breakdown: Dict[str, float]


def generate_reasoning(
    candidate: Dict[str, Any],
    all_scores: Dict[str, Tuple[float, List[str]]],
    honeypot_reasons: List[str],
    disqualifiers: List[str]
) -> str:
    """
    Generate human-readable reasoning for the ranking.

    The reasoning should be:
    - Specific (reference actual candidate data)
    - Honest (acknowledge concerns)
    - JD-connected (reference requirements)

    Args:
        candidate: Candidate dictionary
        all_scores: All computed scores with reasons
        honeypot_reasons: List of honeypot detection reasons
        disqualifiers: List of triggered disqualifiers

    Returns:
        Reasoning string
    """
    parts = []

    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})

    # Lead with title and experience
    parts.append(f"{profile.get('current_title', 'Unknown')} with {profile.get('years_of_experience', 0):.1f} yrs")

    # If honeypot, state it clearly
    if honeypot_reasons:
        parts.append(f"HONEYPOT: {honeypot_reasons[0]}")
        return "; ".join(parts)

    # If major disqualifiers, mention them
    if disqualifiers:
        parts.append(f"Concerns: {', '.join(disqualifiers[:2])}")

    # Add top scoring dimensions (positive signals)
    score_items = [(dim, score, reasons) for dim, (score, reasons) in all_scores.items()]
    score_items.sort(key=lambda x: -x[1])

    for dim, score, reasons in score_items[:3]:
        if reasons and score > 0.3 and reasons[0] not in str(parts):
            parts.append(reasons[0])

    # Behavioral highlights (critical for hiring)
    response_rate = signals.get('recruiter_response_rate', 0)
    notice = signals.get('notice_period_days', 0)
    if response_rate > 0 or notice > 0:
        parts.append(f"Response: {response_rate:.0%}, Notice: {notice}d")

    # Location
    location = profile.get('location', '')
    if location:
        city = location.split(',')[0]
        parts.append(f"Location: {city}")

    return "; ".join(parts[:6])  # Limit to 6 parts for readability


def rank_candidates(
    candidates: List[Dict[str, Any]],
    semantic_scores: Dict[str, float] = None,
    top_n: int = 100,
    verbose: bool = False
) -> List[RankedCandidate]:
    """
    Rank all candidates and return top N.

    Args:
        candidates: List of candidate dictionaries
        semantic_scores: Optional pre-computed semantic scores
        top_n: Number of candidates to return
        verbose: Print progress information

    Returns:
        List of RankedCandidate objects
    """
    if semantic_scores is None:
        semantic_scores = {}

    scored_candidates = []

    for i, candidate in enumerate(candidates):
        if verbose and i % 10000 == 0:
            print(f"Processing candidate {i}/{len(candidates)}...")

        candidate_id = candidate.get('candidate_id', f'UNKNOWN_{i}')

        # Get semantic score (default to 0.5 if not available)
        semantic_score = semantic_scores.get(candidate_id, 0.5)

        # Check for honeypot
        is_honeypot, honeypot_reasons = detect_honeypot(candidate)
        honeypot_penalty = 1.0 if is_honeypot else get_honeypot_score(candidate) * 0.5

        # Check for disqualifiers
        disqualifiers, disqualifier_penalty = detect_disqualifiers(candidate)

        # Compute scores
        final_score, all_scores = compute_final_score(
            candidate,
            semantic_score=semantic_score,
            honeypot_penalty=honeypot_penalty,
            disqualifier_penalty=disqualifier_penalty
        )

        # Generate reasoning
        reasoning = generate_reasoning(
            candidate, all_scores, honeypot_reasons, disqualifiers
        )

        # Store score breakdown
        score_breakdown = {dim: score for dim, (score, _) in all_scores.items()}

        scored_candidates.append({
            'candidate_id': candidate_id,
            'score': round(final_score, 4),  # Round to 4 decimals for consistent sorting/output
            'reasoning': reasoning,
            'is_honeypot': is_honeypot,
            'disqualifiers': disqualifiers,
            'score_breakdown': score_breakdown,
        })

    # Sort by score descending, then by candidate_id ascending for ties
    scored_candidates.sort(key=lambda x: (-x['score'], x['candidate_id']))

    # Create ranked results
    results = []
    for rank, sc in enumerate(scored_candidates[:top_n], start=1):
        results.append(RankedCandidate(
            candidate_id=sc['candidate_id'],
            rank=rank,
            score=sc['score'],
            reasoning=sc['reasoning'],
            is_honeypot=sc['is_honeypot'],
            disqualifiers=sc['disqualifiers'],
            score_breakdown=sc['score_breakdown'],
        ))

    return results


def save_submission(ranked: List[RankedCandidate], output_path: str):
    """
    Save ranking results to CSV in submission format.

    Format: candidate_id,rank,score,reasoning
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])

        for rc in ranked:
            # Escape quotes in reasoning
            reasoning = rc.reasoning.replace('"', "'")
            writer.writerow([
                rc.candidate_id,
                rc.rank,
                f"{rc.score:.4f}",
                reasoning
            ])


def validate_ranking(ranked: List[RankedCandidate]) -> List[str]:
    """
    Validate ranking meets submission requirements.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Check count
    if len(ranked) != 100:
        errors.append(f"Expected 100 candidates, got {len(ranked)}")

    # Check ranks are 1-100
    ranks = [r.rank for r in ranked]
    if set(ranks) != set(range(1, 101)):
        errors.append("Ranks must be exactly 1-100")

    # Check unique candidate IDs
    ids = [r.candidate_id for r in ranked]
    if len(ids) != len(set(ids)):
        errors.append("Duplicate candidate IDs found")

    # Check scores are non-increasing
    scores = [r.score for r in ranked]
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            errors.append(f"Score at rank {i+1} ({scores[i]:.4f}) < rank {i+2} ({scores[i+1]:.4f})")
            break

    return errors


def print_ranking_summary(ranked: List[RankedCandidate]):
    """Print summary of ranking results."""
    print("\n" + "=" * 60)
    print("RANKING SUMMARY")
    print("=" * 60)

    # Count honeypots in top 100
    honeypots = sum(1 for r in ranked if r.is_honeypot)
    print(f"Honeypots in top 100: {honeypots} ({honeypots}%)")

    # Count with disqualifiers
    with_disq = sum(1 for r in ranked if r.disqualifiers)
    print(f"With disqualifiers: {with_disq}")

    # Score distribution
    scores = [r.score for r in ranked]
    print(f"Score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"Avg score: {sum(scores)/len(scores):.4f}")

    # Top 10
    print("\nTop 10:")
    for r in ranked[:10]:
        print(f"  {r.rank}. {r.candidate_id} ({r.score:.4f})")
        print(f"     {r.reasoning[:80]}...")

    # Validation
    errors = validate_ranking(ranked)
    if errors:
        print("\n*** Validation errors: ***")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nValidation: PASSED")
