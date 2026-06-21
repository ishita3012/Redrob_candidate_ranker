"""
Honeypot detection module.

Identifies candidates with impossible or highly suspicious profiles:
- Expert skills with 0 duration
- Skill duration exceeding total experience
- Impossible career timelines

The dataset contains ~80 honeypot candidates designed to trap keyword matchers.
"""

from typing import Dict, Any, List, Tuple
from config import HONEYPOT_THRESHOLDS


def check_expert_zero_duration(candidate: Dict[str, Any]) -> List[str]:
    """
    Check for expert-level skills with 0 months duration.

    This is impossible — you can't be an expert with no time invested.

    Returns:
        List of suspicious skill names
    """
    suspicious = []
    for skill in candidate.get('skills', []):
        if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
            suspicious.append(skill['name'])
    return suspicious


def check_skill_duration_exceeds_experience(candidate: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    """
    Check for skills where duration exceeds total experience.

    Returns:
        List of (skill_name, skill_duration, total_exp_months)
    """
    total_exp_months = candidate.get('profile', {}).get('years_of_experience', 0) * 12
    threshold = HONEYPOT_THRESHOLDS['skill_duration_excess_months']

    suspicious = []
    for skill in candidate.get('skills', []):
        duration = skill.get('duration_months', 0)
        excess = duration - total_exp_months
        if excess > threshold:
            suspicious.append((skill['name'], duration, int(total_exp_months)))

    return suspicious


def check_zero_evidence_skills(candidate: Dict[str, Any]) -> float:
    """
    Calculate ratio of skills with zero evidence.

    Zero evidence = 0 endorsements AND 0/low duration.
    High ratio suggests fake skill listing.

    Returns:
        Ratio of zero-evidence skills (0.0 to 1.0)
    """
    skills = candidate.get('skills', [])
    if not skills:
        return 0.0

    zero_evidence = 0
    for skill in skills:
        if skill.get('endorsements', 0) == 0 and skill.get('duration_months', 0) < 3:
            zero_evidence += 1

    return zero_evidence / len(skills)


def check_impossible_timeline(candidate: Dict[str, Any]) -> bool:
    """
    Check if experience timeline is impossible.

    Example: More years of experience than years since graduation.

    Returns:
        True if timeline is impossible
    """
    profile = candidate.get('profile', {})
    years_exp = profile.get('years_of_experience', 0)

    education = candidate.get('education', [])
    if not education:
        return False

    # Find latest graduation year
    latest_grad = max((e.get('end_year', 0) for e in education), default=0)
    if latest_grad == 0:
        return False

    current_year = 2026  # Competition reference year
    max_possible_exp = current_year - latest_grad + 1  # +1 for buffer

    return years_exp > max_possible_exp


def detect_honeypot(candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Main honeypot detection function.

    Combines multiple checks to determine if a candidate is a honeypot.

    Returns:
        (is_honeypot, list_of_reasons)
    """
    reasons = []
    score = 0

    # Check 1: Expert skills with 0 duration
    expert_zero = check_expert_zero_duration(candidate)
    if len(expert_zero) >= HONEYPOT_THRESHOLDS['expert_zero_duration_count']:
        reasons.append(f"Expert skills with 0 duration: {expert_zero[:3]}")
        score += 3

    # Check 2: Skill duration exceeds experience
    skill_excess = check_skill_duration_exceeds_experience(candidate)
    if skill_excess:
        worst = max(skill_excess, key=lambda x: x[1] - x[2])
        reasons.append(f"Skill '{worst[0]}' duration ({worst[1]}mo) > total exp ({worst[2]}mo)")
        # Score based on severity
        for skill_name, duration, total_exp in skill_excess:
            if duration - total_exp > 36:  # More than 3 years excess
                score += 2
            else:
                score += 1

    # Check 3: High ratio of zero-evidence skills
    zero_ratio = check_zero_evidence_skills(candidate)
    if zero_ratio > HONEYPOT_THRESHOLDS['zero_evidence_skill_ratio']:
        reasons.append(f"{zero_ratio*100:.0f}% of skills have zero evidence")
        score += 2

    # Check 4: Impossible timeline
    if check_impossible_timeline(candidate):
        reasons.append("Experience exceeds possible years since graduation")
        score += 3

    # Determine if honeypot based on accumulated score
    is_honeypot = score >= 3

    return is_honeypot, reasons


def get_honeypot_score(candidate: Dict[str, Any]) -> float:
    """
    Get a suspicion score for honeypot likelihood (0-1).

    Higher = more suspicious.
    Used for soft penalization of borderline cases.

    Returns:
        Suspicion score from 0.0 to 1.0
    """
    score = 0.0

    # Expert skills with 0 duration
    expert_zero = check_expert_zero_duration(candidate)
    score += min(len(expert_zero) * 0.15, 0.45)

    # Skill duration excess
    skill_excess = check_skill_duration_exceeds_experience(candidate)
    for skill_name, duration, total_exp in skill_excess:
        excess_months = duration - total_exp
        if excess_months > 36:
            score += 0.3
        elif excess_months > 24:
            score += 0.2
        else:
            score += 0.1

    # Zero evidence ratio
    zero_ratio = check_zero_evidence_skills(candidate)
    score += zero_ratio * 0.3

    # Impossible timeline
    if check_impossible_timeline(candidate):
        score += 0.4

    return min(score, 1.0)
