"""
JD-specific disqualifier detection.

Based on explicit disqualifiers from the job description:
- Only consulting firm experience
- Title chasers (avg tenure < 18 months)
- Irrelevant titles with AI skills (keyword stuffers)
- No production experience
- Only recent LLM experience
- Research-only background
"""

from typing import Dict, Any, List, Tuple
from config import (
    CONSULTING_FIRMS, IRRELEVANT_TITLES, PENALTIES
)
from data_loader import (
    get_all_companies, get_average_tenure, get_job_count,
    get_career_descriptions, get_skill_names
)


def is_consulting_firm(company: str) -> bool:
    """Check if company is a consulting firm."""
    company_lower = company.lower()
    return any(cf in company_lower for cf in CONSULTING_FIRMS)


def check_only_consulting(candidate: Dict[str, Any]) -> bool:
    """
    Check if candidate has ONLY consulting firm experience.

    JD explicitly says: "People who have only worked at consulting firms
    (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.)"
    """
    companies = get_all_companies(candidate)
    if len(companies) < 1:
        return False

    return all(is_consulting_firm(c) for c in companies)


def check_title_chaser(candidate: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Check if candidate is a title chaser.

    JD says: "If your career trajectory shows you optimizing for titles
    by switching companies every 1.5 years, we're not a fit."

    Returns:
        (is_title_chaser, average_tenure)
    """
    job_count = get_job_count(candidate)
    if job_count < 3:
        return False, 0

    avg_tenure = get_average_tenure(candidate)
    return avg_tenure < 18, avg_tenure


def check_keyword_stuffer(candidate: Dict[str, Any]) -> Tuple[bool, int]:
    """
    Check if candidate is a keyword stuffer.

    Definition: Irrelevant title + 5+ AI/ML skills.
    The JD explicitly warns about this trap.

    Returns:
        (is_stuffer, ai_skill_count)
    """
    current_title = candidate.get('profile', {}).get('current_title', '').lower()

    # Check if title is irrelevant for an AI role
    is_irrelevant = any(t in current_title for t in IRRELEVANT_TITLES)
    if not is_irrelevant:
        return False, 0

    # Count AI/ML skills
    candidate_skills = get_skill_names(candidate)
    ai_skill_keywords = [
        'pytorch', 'tensorflow', 'machine learning', 'deep learning', 'nlp',
        'llm', 'fine-tuning', 'rag', 'embeddings', 'faiss', 'pinecone',
        'langchain', 'hugging face', 'transformers', 'computer vision',
        'gans', 'bert', 'vector search', 'recommendation', 'neural'
    ]

    ai_count = sum(1 for s in candidate_skills
                   if any(ai in s for ai in ai_skill_keywords))

    return ai_count >= 5, ai_count


def check_no_production_experience(candidate: Dict[str, Any]) -> bool:
    """
    Check if candidate has no evidence of production ML experience.

    JD requires: "Production experience with embeddings-based retrieval systems"
    """
    descriptions = get_career_descriptions(candidate)

    production_keywords = [
        'production', 'deployed', 'shipped', 'launched', 'live',
        'real users', 'scale', 'million', 'latency', 'serving',
        'inference', 'api', 'endpoint', 'monitoring'
    ]

    return not any(kw in descriptions for kw in production_keywords)


def check_only_recent_llm_experience(candidate: Dict[str, Any]) -> bool:
    """
    Check if candidate only has recent LLM/LangChain experience.

    JD says: "If your 'AI experience' consists primarily of recent
    (under 12 months) projects using LangChain to call OpenAI —
    we will probably not move forward"
    """
    years_exp = candidate.get('profile', {}).get('years_of_experience', 0)
    if years_exp >= 2:
        return False

    candidate_skills = get_skill_names(candidate)
    llm_skills = ['llm', 'langchain', 'gpt', 'chatgpt', 'openai', 'prompt', 'rag']

    llm_count = sum(1 for s in candidate_skills
                    if any(kw in s for kw in llm_skills))

    # If most of their skills are LLM-related
    if not candidate_skills:
        return False
    return llm_count >= 3 and llm_count / len(candidate_skills) > 0.3


def check_research_only(candidate: Dict[str, Any]) -> bool:
    """
    Check if candidate has only research experience (no production).

    JD says: "If you've spent your career in pure research environments
    without any production deployment — we will not move forward."
    """
    titles = [job.get('title', '').lower() for job in candidate.get('career_history', [])]
    descriptions = get_career_descriptions(candidate)

    # All titles contain "research"
    all_research = all('research' in t for t in titles) if titles else False

    # No production keywords
    no_production = check_no_production_experience(candidate)

    return all_research and no_production


def detect_disqualifiers(candidate: Dict[str, Any]) -> Tuple[List[str], float]:
    """
    Detect all disqualifiers for a candidate.

    Returns:
        (list_of_disqualifiers, total_penalty)
    """
    disqualifiers = []
    penalty = 0.0

    # Check keyword stuffer (biggest trap in the dataset)
    is_stuffer, ai_count = check_keyword_stuffer(candidate)
    if is_stuffer:
        disqualifiers.append(f"keyword_stuffer ({ai_count} AI skills, irrelevant title)")
        penalty = max(penalty, PENALTIES['keyword_stuffer'])

    # Check only consulting
    if check_only_consulting(candidate):
        disqualifiers.append("only_consulting_experience")
        penalty = max(penalty, PENALTIES['only_consulting'])

    # Check title chaser
    is_chaser, avg_tenure = check_title_chaser(candidate)
    if is_chaser:
        disqualifiers.append(f"title_chaser (avg tenure {avg_tenure:.0f}mo)")
        penalty = max(penalty, PENALTIES['title_chaser'])

    # Check no production experience (only if they claim ML skills)
    candidate_skills = get_skill_names(candidate)
    has_ml_skills = any(s in ' '.join(candidate_skills)
                        for s in ['ml', 'machine learning', 'deep learning'])
    if has_ml_skills and check_no_production_experience(candidate):
        disqualifiers.append("no_production_evidence")
        penalty = max(penalty, PENALTIES['no_production'])

    # Check only recent LLM experience
    if check_only_recent_llm_experience(candidate):
        disqualifiers.append("only_recent_llm_experience")
        penalty = max(penalty, 0.5)

    # Check research only
    if check_research_only(candidate):
        disqualifiers.append("research_only_no_production")
        penalty = max(penalty, 0.6)

    return disqualifiers, penalty


def get_disqualifier_penalty(candidate: Dict[str, Any]) -> float:
    """Get total penalty from disqualifiers (0-1)."""
    _, penalty = detect_disqualifiers(candidate)
    return penalty
