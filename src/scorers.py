"""
Multi-dimensional scoring module.

Computes scores across 8 dimensions:
1. Semantic similarity (NEW - embeddings-based)
2. Role fit (title alignment)
3. Skill match (with trust weighting)
4. ML evidence (from career descriptions)
5. Behavioral (availability, responsiveness)
6. Trajectory (career progression)
7. Education
8. GitHub activity
"""

from typing import Dict, Any, List, Tuple
from config import (
    JD_REQUIREMENTS, MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS,
    ML_EVIDENCE_KEYWORDS, BEHAVIORAL_THRESHOLDS, SCORING_WEIGHTS
)
from data_loader import (
    get_skill_names, get_skill_by_name, get_career_descriptions,
    get_all_titles, get_average_tenure, get_job_count, days_since
)


# =============================================================================
# ROLE FIT SCORING
# =============================================================================

def score_role_fit(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score how well candidate's role/title aligns with target.

    Checks:
    - Current title is ML-relevant
    - Career history contains ML roles
    - Experience years in range
    - Location preferences

    Returns:
        (score, list of reasons)
    """
    score = 0.0
    reasons = []

    profile = candidate.get('profile', {})
    current_title = profile.get('current_title', '').lower()

    # Check if current title is ML-relevant
    target_keywords = JD_REQUIREMENTS['target_title_keywords']
    if any(kw in current_title for kw in target_keywords):
        score += 0.4
        reasons.append(f"ML-relevant title: {profile.get('current_title')}")

    # Check for adjacent titles (can also be good)
    adjacent = JD_REQUIREMENTS['adjacent_titles']
    if any(kw in current_title for kw in adjacent):
        score += 0.2
        reasons.append(f"Adjacent tech title: {profile.get('current_title')}")

    # Check career history for ML roles
    all_titles = get_all_titles(candidate)
    ml_roles = sum(1 for t in all_titles
                   if any(kw in t.lower() for kw in target_keywords))
    if ml_roles > 0:
        score += min(ml_roles * 0.1, 0.3)
        reasons.append(f"{ml_roles} ML roles in history")

    # Experience years fit
    years_exp = profile.get('years_of_experience', 0)
    exp_min, exp_max = JD_REQUIREMENTS['experience_range']

    if exp_min <= years_exp <= exp_max:
        score += 0.2
        reasons.append(f"Experience in range ({years_exp:.1f} yrs)")
    elif years_exp < JD_REQUIREMENTS['experience_hard_min']:
        score -= 0.1
        reasons.append(f"Underqualified ({years_exp:.1f} yrs)")
    elif years_exp > JD_REQUIREMENTS['experience_hard_max']:
        score -= 0.05
        reasons.append(f"Possibly overqualified ({years_exp:.1f} yrs)")

    # Location fit
    location = profile.get('location', '').lower()
    if any(loc in location for loc in JD_REQUIREMENTS['preferred_locations']):
        score += 0.1
        reasons.append("Preferred location")

    return min(max(score, 0), 1.0), reasons


# =============================================================================
# SKILL MATCH SCORING
# =============================================================================

def compute_skill_trust(skill: Dict[str, Any], candidate: Dict[str, Any]) -> float:
    """
    Compute trust score for a single skill based on evidence.

    Trust factors:
    - Duration of skill usage
    - Number of endorsements
    - Proficiency level claimed
    - Platform assessment scores
    - Mentioned in career descriptions
    """
    trust = 0.1  # Base: they listed it

    # Duration adds trust
    duration = skill.get('duration_months', 0)
    trust += min(duration / 48, 0.2)  # Max 0.2 for 4+ years

    # Endorsements add trust
    endorsements = skill.get('endorsements', 0)
    trust += min(endorsements / 30, 0.15)  # Max 0.15 for 30+ endorsements

    # Proficiency level
    proficiency = skill.get('proficiency', '')
    if proficiency == 'expert':
        trust += 0.15
    elif proficiency == 'advanced':
        trust += 0.1
    elif proficiency == 'intermediate':
        trust += 0.05

    # Platform assessment (gold standard - verified by platform)
    assessments = candidate.get('redrob_signals', {}).get('skill_assessment_scores', {})
    skill_name = skill.get('name', '')
    if skill_name in assessments:
        assessment_score = assessments[skill_name]
        trust += (assessment_score / 100) * 0.3  # Max 0.3 for 100%

    # Check if mentioned in career descriptions (evidence of actual use)
    descriptions = get_career_descriptions(candidate)
    if skill_name.lower() in descriptions:
        trust += 0.1

    return min(trust, 1.0)


def score_skill_match(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score skill match against required skills with trust weighting.

    Doesn't just check if skill is listed - verifies it through
    duration, endorsements, assessments, and career evidence.

    Returns:
        (score, list of reasons)
    """
    candidate_skills = get_skill_names(candidate)
    reasons = []

    must_have_matches = 0
    must_have_trust_sum = 0.0
    nice_to_have_matches = 0

    # Check must-have skills
    for req_skill in MUST_HAVE_SKILLS:
        req_lower = req_skill.lower()
        matched = None
        for cs in candidate_skills:
            if req_lower in cs or cs in req_lower:
                # Find the full skill object
                for s in candidate.get('skills', []):
                    if s['name'].lower() == cs:
                        matched = s
                        break
                break

        if matched:
            trust = compute_skill_trust(matched, candidate)
            must_have_matches += 1
            must_have_trust_sum += trust

    # Check nice-to-have skills
    for req_skill in NICE_TO_HAVE_SKILLS:
        req_lower = req_skill.lower()
        for cs in candidate_skills:
            if req_lower in cs or cs in req_lower:
                nice_to_have_matches += 1
                break

    # Calculate score
    if must_have_matches > 0:
        avg_trust = must_have_trust_sum / must_have_matches
        coverage = must_have_matches / len(MUST_HAVE_SKILLS)
        score = (coverage * 0.6) + (avg_trust * 0.4)
    else:
        score = 0.0

    # Bonus for nice-to-have
    score += min(nice_to_have_matches * 0.03, 0.15)

    if must_have_matches > 0:
        reasons.append(f"{must_have_matches} must-have skills (avg trust: {must_have_trust_sum/must_have_matches:.2f})")
    if nice_to_have_matches > 0:
        reasons.append(f"{nice_to_have_matches} nice-to-have skills")

    return min(score, 1.0), reasons


# =============================================================================
# ML EVIDENCE SCORING
# =============================================================================

def score_ml_evidence(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score based on ML/production evidence in career descriptions.

    This catches hidden gems — candidates with non-ML titles
    but actual ML work in their job descriptions.

    Returns:
        (score, list of reasons)
    """
    descriptions = get_career_descriptions(candidate)
    reasons = []

    evidence_count = 0
    found_keywords = []

    for kw in ML_EVIDENCE_KEYWORDS:
        if kw in descriptions:
            evidence_count += 1
            if len(found_keywords) < 5:
                found_keywords.append(kw)

    # Score based on evidence count
    score = min(evidence_count * 0.08, 0.8)

    # Bonus for specific high-value evidence
    high_value = ['embedding', 'vector search', 'ranking', 'retrieval',
                  'recommendation', 'deployed', 'production']
    hv_count = sum(1 for hv in high_value if hv in descriptions)
    score += min(hv_count * 0.05, 0.2)

    if found_keywords:
        reasons.append(f"ML evidence: {', '.join(found_keywords[:3])}")

    return min(score, 1.0), reasons


# =============================================================================
# BEHAVIORAL SCORING
# =============================================================================

def score_behavioral(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score behavioral signals (availability, responsiveness).

    The JD says: "A perfect-on-paper candidate who hasn't logged in
    for 6 months and has a 5% response rate is NOT actually available."

    This is 25% of the total score — very important!

    Returns:
        (score, list of reasons)
    """
    signals = candidate.get('redrob_signals', {})
    reasons = []
    score = 0.0

    # Response rate (how often they reply to recruiters)
    response_rate = signals.get('recruiter_response_rate', 0)
    if response_rate >= BEHAVIORAL_THRESHOLDS['response_rate_good']:
        score += 0.25
        reasons.append(f"High response rate ({response_rate:.0%})")
    elif response_rate >= BEHAVIORAL_THRESHOLDS['response_rate_okay']:
        score += 0.15
    elif response_rate < BEHAVIORAL_THRESHOLDS['response_rate_bad']:
        score -= 0.1
        reasons.append(f"Low response rate ({response_rate:.0%})")

    # Last active (when did they last log in)
    days_inactive = days_since(signals.get('last_active_date', ''))
    if days_inactive <= BEHAVIORAL_THRESHOLDS['days_inactive_good']:
        score += 0.2
        reasons.append("Recently active")
    elif days_inactive <= BEHAVIORAL_THRESHOLDS['days_inactive_okay']:
        score += 0.1
    elif days_inactive > BEHAVIORAL_THRESHOLDS['days_inactive_bad']:
        score -= 0.2
        reasons.append(f"Inactive {days_inactive} days")

    # Open to work flag
    if signals.get('open_to_work_flag'):
        score += 0.15
        reasons.append("Open to work")

    # Notice period
    notice = signals.get('notice_period_days', 90)
    if notice <= JD_REQUIREMENTS['preferred_notice_days']:
        score += 0.15
        reasons.append(f"Short notice ({notice}d)")
    elif notice <= JD_REQUIREMENTS['acceptable_notice_days']:
        score += 0.05
    else:
        score -= 0.05
        reasons.append(f"Long notice ({notice}d)")

    # Willing to relocate
    if signals.get('willing_to_relocate'):
        score += 0.1
        reasons.append("Will relocate")

    # Interview completion rate
    interview_rate = signals.get('interview_completion_rate', 0)
    if interview_rate >= BEHAVIORAL_THRESHOLDS['interview_completion_good']:
        score += 0.1
    elif interview_rate < BEHAVIORAL_THRESHOLDS['interview_completion_okay']:
        score -= 0.05

    # Verified contact info
    if signals.get('verified_email') and signals.get('verified_phone'):
        score += 0.05

    return min(max(score, 0), 1.0), reasons


# =============================================================================
# CAREER TRAJECTORY SCORING
# =============================================================================

def score_trajectory(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score career trajectory (progression, stability, company quality).

    Returns:
        (score, list of reasons)
    """
    reasons = []
    score = 0.0

    # Tenure stability
    avg_tenure = get_average_tenure(candidate)
    job_count = get_job_count(candidate)

    if avg_tenure >= 24:
        score += 0.3
        reasons.append(f"Stable tenure ({avg_tenure:.0f}mo avg)")
    elif avg_tenure >= 18:
        score += 0.2
    elif avg_tenure < 12 and job_count >= 3:
        score -= 0.2
        reasons.append("Short tenures")

    # Check for progression (title seniority)
    titles = get_all_titles(candidate)
    if titles:
        seniority_keywords = ['senior', 'lead', 'staff', 'principal', 'director', 'head']
        current_title = titles[0].lower() if titles else ''

        if any(kw in current_title for kw in seniority_keywords):
            score += 0.2
            reasons.append("Senior-level position")

    # Company quality (product companies vs consulting)
    from disqualifiers import is_consulting_firm
    companies = [job['company'] for job in candidate.get('career_history', [])]
    product_companies = [c for c in companies if not is_consulting_firm(c)]

    if len(product_companies) >= 2:
        score += 0.2
        reasons.append(f"{len(product_companies)} product companies")

    # Industry relevance
    industries = [job.get('industry', '').lower() for job in candidate.get('career_history', [])]
    relevant_industries = ['ai', 'ml', 'tech', 'software', 'internet', 'fintech', 'e-commerce']
    relevant_count = sum(1 for ind in industries if any(ri in ind for ri in relevant_industries))
    score += min(relevant_count * 0.05, 0.2)

    return min(max(score, 0), 1.0), reasons


# =============================================================================
# EDUCATION SCORING
# =============================================================================

def score_education(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score education (tier, field relevance, degree level).

    Returns:
        (score, list of reasons)
    """
    education = candidate.get('education', [])
    if not education:
        return 0.3, ["No education data"]  # Neutral

    reasons = []
    score = 0.0

    # Best tier
    tiers = [e.get('tier', 'unknown') for e in education]
    tier_scores = {'tier_1': 0.4, 'tier_2': 0.3, 'tier_3': 0.2, 'tier_4': 0.1, 'unknown': 0.15}
    best_tier_score = max(tier_scores.get(t, 0.1) for t in tiers)
    score += best_tier_score

    best_tier = min(tiers, key=lambda t: ['tier_1', 'tier_2', 'tier_3', 'tier_4', 'unknown'].index(t)
                    if t in ['tier_1', 'tier_2', 'tier_3', 'tier_4', 'unknown'] else 5)
    if best_tier in ['tier_1', 'tier_2']:
        reasons.append(f"Education: {best_tier}")

    # Field relevance
    relevant_fields = ['computer', 'software', 'data', 'machine learning', 'ai',
                       'statistics', 'mathematics', 'electrical', 'information']
    fields = [e.get('field_of_study', '').lower() for e in education]

    for field in fields:
        if any(rf in field for rf in relevant_fields):
            score += 0.2
            reasons.append(f"Relevant field: {field[:30]}")
            break

    # Higher degree
    degrees = [e.get('degree', '').lower() for e in education]
    if any('phd' in d or 'doctorate' in d for d in degrees):
        score += 0.2
        reasons.append("PhD")
    elif any('master' in d or 'm.s' in d or 'm.tech' in d for d in degrees):
        score += 0.1
        reasons.append("Master's degree")

    return min(score, 1.0), reasons


# =============================================================================
# GITHUB SCORING
# =============================================================================

def score_github(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Score GitHub activity.

    Uses the github_activity_score from redrob_signals.

    Returns:
        (score, list of reasons)
    """
    signals = candidate.get('redrob_signals', {})
    github_score = signals.get('github_activity_score', -1)
    reasons = []

    if github_score < 0:
        return 0.0, ["No GitHub"]

    # Normalize to 0-1
    normalized = min(github_score / 100, 1.0)
    score = normalized * 0.8  # Max 0.8 from GitHub alone

    if github_score >= 50:
        reasons.append(f"Strong GitHub ({github_score:.0f})")
    elif github_score >= 20:
        reasons.append(f"Active GitHub ({github_score:.0f})")

    return score, reasons


# =============================================================================
# COMBINED SCORING
# =============================================================================

def compute_all_scores(
    candidate: Dict[str, Any],
    semantic_score: float = 0.5
) -> Dict[str, Tuple[float, List[str]]]:
    """
    Compute all scoring dimensions.

    Args:
        candidate: Candidate dictionary
        semantic_score: Pre-computed semantic similarity score (0-1)

    Returns:
        Dictionary of {dimension: (score, reasons)}
    """
    return {
        'semantic': (semantic_score, [f"Semantic similarity: {semantic_score:.2f}"]),
        'role_fit': score_role_fit(candidate),
        'skill_match': score_skill_match(candidate),
        'ml_evidence': score_ml_evidence(candidate),
        'behavioral': score_behavioral(candidate),
        'trajectory': score_trajectory(candidate),
        'education': score_education(candidate),
        'github': score_github(candidate),
    }


def compute_final_score(
    candidate: Dict[str, Any],
    semantic_score: float = 0.5,
    honeypot_penalty: float = 0.0,
    disqualifier_penalty: float = 0.0
) -> Tuple[float, Dict]:
    """
    Compute final weighted score.

    Args:
        candidate: Candidate dictionary
        semantic_score: Pre-computed semantic similarity score
        honeypot_penalty: Penalty for honeypot indicators (0-1)
        disqualifier_penalty: Penalty for JD disqualifiers (0-1)

    Returns:
        (final_score, score_breakdown)
    """
    all_scores = compute_all_scores(candidate, semantic_score)

    # Weighted sum
    weighted_score = 0.0
    for dimension, weight in SCORING_WEIGHTS.items():
        score, _ = all_scores.get(dimension, (0.0, []))
        weighted_score += score * weight

    # Apply penalties
    if honeypot_penalty > 0:
        weighted_score *= (1 - honeypot_penalty)

    if disqualifier_penalty > 0:
        weighted_score *= (1 - disqualifier_penalty)

    return weighted_score, all_scores
