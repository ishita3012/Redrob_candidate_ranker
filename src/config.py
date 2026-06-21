"""
Configuration for the Intelligent Candidate Ranker.

All JD-derived requirements, scoring weights, and thresholds are defined here.
This makes it easy to tune the system without changing code.
"""

from typing import Dict, List, Set

# =============================================================================
# JD-DERIVED REQUIREMENTS
# =============================================================================

JD_REQUIREMENTS: Dict = {
    # Target role keywords (from job_description.docx)
    "target_title_keywords": [
        "ml engineer", "machine learning", "ai engineer", "data scientist",
        "research engineer", "nlp engineer", "deep learning", "applied ml",
        "recommendation", "search engineer", "mlops", "computer vision"
    ],

    # Adjacent titles that could be good fits
    "adjacent_titles": [
        "backend engineer", "software engineer", "data engineer",
        "analytics engineer", "platform engineer", "staff engineer"
    ],

    # Experience requirements
    "experience_range": (5, 9),      # Preferred: 5-9 years
    "experience_hard_min": 3,        # Below this is too junior
    "experience_hard_max": 15,       # Above this might not be coding

    # Location preferences (India focus)
    "preferred_locations": [
        "pune", "noida", "hyderabad", "bangalore", "bengaluru",
        "mumbai", "delhi", "gurgaon", "gurugram", "chennai"
    ],

    # Notice period
    "preferred_notice_days": 30,     # JD prefers <30 days
    "acceptable_notice_days": 60,    # Up to 60 is okay
}

# =============================================================================
# SKILLS CONFIGURATION
# =============================================================================

# Must-have skills from JD (production embeddings, vector DBs, ranking)
MUST_HAVE_SKILLS: List[str] = [
    # Embeddings & Retrieval
    "embeddings", "sentence transformers", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "vector search", "semantic search", "information retrieval",

    # ML/AI Core
    "pytorch", "tensorflow", "machine learning", "deep learning", "nlp",
    "transformers", "hugging face", "bert", "llm", "fine-tuning",

    # Ranking & Evaluation
    "ranking", "recommendation systems", "learning to rank", "ndcg", "mrr",
    "a/b testing", "evaluation", "metrics",

    # Production ML
    "mlops", "mlflow", "kubeflow", "model deployment", "inference",
]

# Nice-to-have skills
NICE_TO_HAVE_SKILLS: List[str] = [
    "lora", "qlora", "peft", "rag", "langchain",
    "xgboost", "lightgbm", "feature engineering",
    "distributed systems", "kubernetes", "docker",
]

# =============================================================================
# DISQUALIFIER CONFIGURATION
# =============================================================================

# Consulting firms (JD explicitly disqualifies only-consulting experience)
CONSULTING_FIRMS: Set[str] = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tech mahindra", "hcl", "mindtree", "lti", "mphasis", "niit",
    "ibm global services", "deloitte", "ey", "pwc", "kpmg", "genpact",
    "cts", "cognizant technology", "infosys bpm", "wipro technologies"
}

# Irrelevant titles (keyword stuffers have these + AI skills)
IRRELEVANT_TITLES: Set[str] = {
    "marketing manager", "hr manager", "accountant", "sales executive",
    "content writer", "graphic designer", "customer support",
    "civil engineer", "mechanical engineer", "operations manager",
    "business analyst", "project manager"
}

# =============================================================================
# ML EVIDENCE KEYWORDS
# =============================================================================

# Keywords to search for in career descriptions (proof of work)
ML_EVIDENCE_KEYWORDS: List[str] = [
    # Production ML
    "deployed model", "production ml", "model serving", "inference pipeline",
    "training pipeline", "feature store", "model registry",

    # Retrieval & Search
    "embedding", "vector search", "retrieval", "ranking model", "recommendation",
    "search relevance", "semantic search", "information retrieval",
    "learning to rank", "reranking", "candidate generation",

    # Specific technologies
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
    "pytorch", "tensorflow", "transformers", "hugging face", "bert",

    # Evaluation
    "a/b test", "ndcg", "mrr", "precision", "recall", "f1",
    "offline evaluation", "online evaluation",

    # LLM/RAG
    "rag", "retrieval augmented", "fine-tun", "llm", "prompt engineering",

    # General ML
    "machine learning", "deep learning", "neural network", "nlp pipeline",
]

# =============================================================================
# SCORING WEIGHTS
# =============================================================================

# These weights determine how much each dimension contributes to the final score.
# Sum should equal 1.0 for easy interpretation.

SCORING_WEIGHTS: Dict[str, float] = {
    "semantic": 0.15,        # Semantic similarity to JD (NEW)
    "role_fit": 0.15,        # Title and career trajectory alignment
    "skill_match": 0.15,     # Verified skills match
    "ml_evidence": 0.15,     # Evidence in career descriptions
    "behavioral": 0.25,      # Can we actually hire them? (HIGHEST)
    "trajectory": 0.10,      # Career progression
    "education": 0.03,       # Education tier
    "github": 0.02,          # GitHub activity
}

# Verify weights sum to 1.0
assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 0.001, "Weights must sum to 1.0"

# =============================================================================
# BEHAVIORAL THRESHOLDS
# =============================================================================

BEHAVIORAL_THRESHOLDS: Dict[str, float] = {
    "response_rate_good": 0.7,
    "response_rate_okay": 0.4,
    "response_rate_bad": 0.2,

    "days_inactive_good": 14,
    "days_inactive_okay": 60,
    "days_inactive_bad": 180,

    "interview_completion_good": 0.8,
    "interview_completion_okay": 0.5,
}

# =============================================================================
# HONEYPOT DETECTION THRESHOLDS
# =============================================================================

HONEYPOT_THRESHOLDS: Dict[str, int] = {
    "skill_duration_excess_months": 24,   # Skill duration > exp by this much
    "expert_zero_duration_count": 3,      # Expert skills with 0 months
    "zero_evidence_skill_ratio": 0.7,     # % of skills with no evidence
}

# =============================================================================
# PENALTY WEIGHTS
# =============================================================================

# How much to penalize each disqualifier (0-1, where 1 = complete disqualification)
PENALTIES: Dict[str, float] = {
    "honeypot": 1.0,              # Complete disqualification
    "keyword_stuffer": 0.95,      # Almost complete (irrelevant title + AI skills)
    "only_consulting": 0.7,       # Heavy penalty
    "title_chaser": 0.4,          # Moderate penalty
    "no_production": 0.5,         # Moderate penalty
    "inactive_6_months": 0.4,     # Moderate penalty
    "low_response_rate": 0.2,     # Light penalty
}

# =============================================================================
# SEMANTIC MATCHING CONFIGURATION
# =============================================================================

SEMANTIC_CONFIG: Dict = {
    # Model to use for embeddings
    "model_name": "all-MiniLM-L6-v2",  # Fast, good quality, 384 dimensions

    # Alternative models (uncomment to use):
    # "model_name": "all-mpnet-base-v2",  # Higher quality, slower
    # "model_name": "paraphrase-MiniLM-L6-v2",  # Good for paraphrase detection

    # Pre-computation settings
    "batch_size": 64,
    "embeddings_file": "candidate_embeddings.npy",
    "index_file": "candidate_index.faiss",
}

# JD summary for semantic matching (extracted key requirements)
JD_SUMMARY: str = """
Senior AI Engineer for talent intelligence platform.
Requirements: 5-9 years experience building production ML systems.
Must have: embeddings, vector databases, retrieval systems, ranking models.
Must have: PyTorch or TensorFlow, evaluation frameworks (NDCG, MRR).
Looking for: shipped ML to real users, hybrid retrieval, LLM integration.
Culture: scrappy, ships fast, writes async, disagrees openly.
Red flags: pure research only, only consulting, title chasing, only recent LLM experience.
Location: Pune or Noida preferred, India cities acceptable.
"""
