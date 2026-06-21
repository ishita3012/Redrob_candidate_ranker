"""
Data loading and preprocessing utilities.
"""

import json
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional


def load_candidates(filepath: str) -> List[Dict[str, Any]]:
    """
    Load candidates from JSONL file.

    Args:
        filepath: Path to candidates.jsonl file

    Returns:
        List of candidate dictionaries
    """
    candidates = []
    path = Path(filepath)

    # Handle both .jsonl and .json files
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                candidates = data
            else:
                candidates = [data]
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))

    return candidates


def parse_date(date_str: str) -> Optional[date]:
    """Parse date string to date object."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return None


def days_since(date_str: str, reference_date: date = None) -> int:
    """
    Calculate days since a given date.

    Args:
        date_str: Date string in YYYY-MM-DD format
        reference_date: Reference date (defaults to competition date)

    Returns:
        Number of days since the date (999 if unknown)
    """
    if reference_date is None:
        reference_date = date(2026, 6, 21)  # Competition reference date

    parsed = parse_date(date_str)
    if parsed is None:
        return 999  # Unknown = assume very old

    return (reference_date - parsed).days


def extract_text_for_embedding(candidate: Dict[str, Any]) -> str:
    """
    Extract all relevant text from candidate for embedding.

    Combines headline, summary, job descriptions, and skills into
    a single text string for semantic embedding.
    """
    parts = []

    # Profile information
    profile = candidate.get('profile', {})
    parts.append(profile.get('headline', ''))
    parts.append(profile.get('summary', ''))
    parts.append(profile.get('current_title', ''))

    # Career history descriptions (rich source of information)
    for job in candidate.get('career_history', []):
        parts.append(job.get('title', ''))
        parts.append(job.get('description', ''))

    # Skills (just names)
    skills = [s['name'] for s in candidate.get('skills', [])]
    parts.append(' '.join(skills))

    # Certifications
    certs = [c['name'] for c in candidate.get('certifications', [])]
    parts.append(' '.join(certs))

    return ' '.join(filter(None, parts))


def get_all_companies(candidate: Dict[str, Any]) -> List[str]:
    """Get all companies from career history."""
    return [job['company'] for job in candidate.get('career_history', [])]


def get_all_titles(candidate: Dict[str, Any]) -> List[str]:
    """Get all titles from career history."""
    return [job['title'] for job in candidate.get('career_history', [])]


def get_total_experience_months(candidate: Dict[str, Any]) -> float:
    """Get total experience in months."""
    return candidate.get('profile', {}).get('years_of_experience', 0) * 12


def get_skill_names(candidate: Dict[str, Any]) -> List[str]:
    """Get all skill names (lowercase)."""
    return [s['name'].lower() for s in candidate.get('skills', [])]


def get_skill_by_name(candidate: Dict[str, Any], skill_name: str) -> Optional[Dict]:
    """Get skill details by name (case-insensitive)."""
    skill_lower = skill_name.lower()
    for s in candidate.get('skills', []):
        if s['name'].lower() == skill_lower:
            return s
    return None


def get_career_descriptions(candidate: Dict[str, Any]) -> str:
    """Get concatenated career descriptions (lowercase)."""
    descriptions = []
    for job in candidate.get('career_history', []):
        descriptions.append(job.get('description', ''))
    return ' '.join(descriptions).lower()


def get_average_tenure(candidate: Dict[str, Any]) -> float:
    """Calculate average tenure across all jobs in months."""
    tenures = [job['duration_months'] for job in candidate.get('career_history', [])]
    if not tenures:
        return 0
    return sum(tenures) / len(tenures)


def get_job_count(candidate: Dict[str, Any]) -> int:
    """Get number of jobs in career history."""
    return len(candidate.get('career_history', []))


def create_candidate_id_map(candidates: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Create a mapping from candidate_id to candidate data."""
    return {c['candidate_id']: c for c in candidates}
