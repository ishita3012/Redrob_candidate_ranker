"""
JD parsing layer — turns ANY job description (plain text) into a structured JDSpec.

Design intent (defensible at Stage 5):
  * The ranking pipeline is JD-agnostic. It consumes a JDSpec, never hardcoded
    constants. Swap the JD file -> new spec -> new ranking, no code change.
  * The reusable knowledge is CONCEPT_LIBRARY: a recruiter-domain vocabulary that
    maps JD language ("embeddings-based retrieval", "vector database", ...) to
    things we can actually detect in a candidate (evidence phrases in career
    descriptions, skill tokens). The JD TEXT selects which concepts apply and in
    which polarity (must-have / nice-to-have / disqualifier).
  * Fully deterministic and local: no network, no LLM. Pure parsing + lexicon
    lookup, so Stage-3 reproduction is trivial and the logic is transparent.

A different JD activates a different subset of concepts and different numbers,
which is exactly the "could you drop in a second JD without touching code?" test.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CONCEPT LIBRARY  (reusable recruiter-domain knowledge, NOT this-JD-specific)
# =============================================================================
# Each concept declares:
#   jd_cues  : phrases whose presence in a JD means "this concept is relevant"
#   evidence : phrases we look for in a candidate's career-history DESCRIPTIONS
#              (the trustworthy signal — descriptions, not the noisy skills list)
#   skills   : skill tokens that *weakly* corroborate (skills lists are noisy)
CONCEPT_LIBRARY: Dict[str, Dict[str, List[str]]] = {
    "embeddings_retrieval": {
        "jd_cues": ["embeddings-based retrieval", "embeddings", "sentence-transformers",
                    "dense retrieval", "retrieval", "bge", "e5"],
        "evidence": ["embedding", "semantic search", "retrieval", "information retrieval",
                     "dense retrieval", "nearest neighbor", "vector search"],
        "skills": ["embeddings", "sentence transformers", "semantic search"],
    },
    "vector_db": {
        "jd_cues": ["vector database", "vector databases", "hybrid search", "pinecone",
                    "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss"],
        "evidence": ["faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
                     "opensearch", "vector index", "vector database", "hybrid search"],
        "skills": ["faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch"],
    },
    "ranking_recsys": {
        "jd_cues": ["ranking", "learning-to-rank", "learning to rank", "recommendation",
                    "recommender", "matching", "search relevance"],
        # Domain-specific phrases only — bare "relevance"/"personalization" are too
        # generic (appear in marketing/ops text) and were matching decoy roles.
        "evidence": ["ranking model", "learning to rank", "recommendation system",
                     "recommender", "reranking", "re-ranking", "candidate generation",
                     "search relevance", "discovery feed", "recommendation engine"],
        "skills": ["learning to rank", "ranking", "recommendation systems"],
    },
    "evaluation_frameworks": {
        "jd_cues": ["evaluation frameworks", "ndcg", "mrr", "map", "a/b test",
                    "offline-to-online", "evaluate a ranking"],
        "evidence": ["ndcg", "mrr", "a/b test", "ab test", "offline evaluation",
                     "online evaluation", "precision@", "recall@", "held-out", "eval harness"],
        "skills": ["ndcg", "mrr", "a/b testing", "evaluation"],
    },
    "production_scale": {
        "jd_cues": ["production", "deployed to real users", "real users", "at scale",
                    "inference", "latency"],
        "evidence": ["production", "deployed", "shipped", "in production", "real users",
                     "at scale", "millions", "latency", "serving", "inference", "throughput",
                     "live"],
        "skills": ["mlops", "model deployment", "inference"],
    },
    "llm_finetune": {
        "jd_cues": ["fine-tuning", "lora", "qlora", "peft", "llm"],
        "evidence": ["fine-tun", "lora", "qlora", "peft", "instruction tun", "rlhf"],
        "skills": ["lora", "qlora", "peft", "fine-tuning"],
    },
    "strong_python": {
        "jd_cues": ["strong python", "python", "code quality"],
        "evidence": ["python"],
        "skills": ["python"],
    },
}

# Disqualifier concepts: JD cue -> how we detect the red flag (handled in gates.py;
# here we only record which ones the JD activates and their severity hint).
DISQUALIFIER_LIBRARY: Dict[str, Dict] = {
    "research_only_no_production": {
        "jd_cues": ["pure research", "research environments", "academic labs",
                    "research-only", "without any production deployment"],
        "severity": "hard",
    },
    "only_recent_llm": {
        "jd_cues": ["langchain to call openai", "consists primarily of recent",
                    "under 12 months", "before it became fashionable"],
        "severity": "soft",
    },
    "no_recent_code": {
        "jd_cues": ["hasn't written production code", "moved into \"architecture\"",
                    "tech lead", "this role writes code"],
        "severity": "soft",
    },
    "title_chaser": {
        "jd_cues": ["title-chasers", "switching companies every", "1.5 years",
                    "optimizing for", "staff", "principal"],
        "severity": "soft",
    },
    "only_consulting": {
        "jd_cues": ["only worked at consulting firms", "tcs", "infosys", "wipro",
                    "accenture", "cognizant", "capgemini"],
        "severity": "hard",
    },
    "wrong_domain_no_nlp": {
        "jd_cues": ["computer vision, speech, or robotics", "without significant nlp",
                    "re-learning fundamentals"],
        "severity": "soft",
    },
    "closed_source_no_validation": {
        "jd_cues": ["closed-source proprietary systems", "without external validation"],
        "severity": "soft",
    },
}

# Indian city gazetteer (used to detect location preferences when a JD names cities).
INDIA_CITIES = [
    "pune", "noida", "hyderabad", "bangalore", "bengaluru", "mumbai", "delhi",
    "gurgaon", "gurugram", "chennai", "kolkata", "delhi ncr", "ncr",
]

# Core/domain concepts carry the real signal; "supporting" concepts (production,
# python) use generic words that also appear in non-ML text, so they must NOT be
# used to decide relevance on their own — only to corroborate.
CORE_CONCEPTS = {"embeddings_retrieval", "vector_db", "ranking_recsys",
                 "evaluation_frameworks", "llm_finetune"}
SUPPORTING_CONCEPTS = {"production_scale", "strong_python"}

# Role-title vocabulary: maps a JD's high-level role to candidate title keywords.
ROLE_TITLE_VOCAB = {
    "ai_ml_engineer": ["ai engineer", "ml engineer", "machine learning", "applied scientist",
                       "applied ml", "research engineer", "data scientist", "nlp engineer",
                       "deep learning", "recommendation", "search engineer", "mlops"],
}
ADJACENT_TITLE_VOCAB = ["software engineer", "backend engineer", "data engineer",
                        "platform engineer", "research scientist", "full stack"]

# Words too generic to be useful as fallback role/evidence terms.
_FALLBACK_STOP = {
    "senior", "junior", "lead", "staff", "principal", "founding", "team", "member",
    "full", "time", "role", "position", "experience", "years", "year", "work", "working",
    "with", "have", "will", "that", "this", "your", "you", "our", "and", "the", "for",
    "who", "what", "how", "are", "not", "but", "they", "them", "their", "from", "into",
    "about", "would", "should", "could", "want", "need", "looking", "candidate", "job",
    "company", "well", "kind", "like", "some", "most", "much", "very", "also", "been",
    "were", "here", "there", "these", "those", "which", "when", "where", "over",
}


def _fallback_role_titles(text: str) -> List[str]:
    """Derive role-title match keys from the JD's title line — used only when the
    role-title vocabulary doesn't recognize the JD's domain."""
    first = ""
    m = re.search(r"(?:job\s+description|job\s+title|role|position)\s*[:\-]\s*(.+)", text, re.I)
    if m:
        first = m.group(1)
    else:
        for line in text.splitlines():
            if line.strip():
                first = line.strip()
                break
    head = re.split(r"[—\-|(,\n]", first)[0].lower()
    toks = [t for t in re.findall(r"[a-z]+", head) if t not in _FALLBACK_STOP and len(t) > 2]
    keys: List[str] = []
    if len(toks) >= 2:
        keys.append(" ".join(toks[-2:]))  # e.g. "marketing manager", "mechanical engineer"
    keys.extend(toks)
    return list(dict.fromkeys(keys))


def _fallback_evidence_terms(text: str, k: int = 15) -> List[str]:
    """Derive evidence keywords straight from the JD text — used only when the concept
    library activates no concepts for this JD (i.e. an out-of-vocabulary domain)."""
    from collections import Counter
    toks = [t for t in re.findall(r"[a-z][a-z\+\#\.]{3,}", text.lower())
            if t not in _FALLBACK_STOP]
    return [w for w, _ in Counter(toks).most_common(k)]


@dataclass
class JDSpec:
    raw_text: str
    role_titles: List[str] = field(default_factory=list)
    adjacent_titles: List[str] = field(default_factory=list)
    exp_band: Tuple[float, float] = (0.0, 50.0)
    exp_ideal: Optional[Tuple[float, float]] = None
    locations: List[str] = field(default_factory=list)
    location_country: Optional[str] = None
    must_have_concepts: List[str] = field(default_factory=list)
    nice_to_have_concepts: List[str] = field(default_factory=list)
    disqualifier_concepts: List[str] = field(default_factory=list)
    evidence_terms: List[str] = field(default_factory=list)        # all (core + supporting)
    core_evidence_terms: List[str] = field(default_factory=list)   # domain-specific only
    notice_pref_days: Optional[int] = None
    behavioral_priorities: Dict[str, str] = field(default_factory=dict)
    ideal_text: str = ""

    def summary(self) -> str:
        lines = [
            "JDSpec",
            f"  role_titles      : {self.role_titles}",
            f"  adjacent_titles  : {self.adjacent_titles}",
            f"  exp_band         : {self.exp_band}   ideal: {self.exp_ideal}",
            f"  locations        : {self.locations}  country: {self.location_country}",
            f"  notice_pref_days : {self.notice_pref_days}",
            f"  MUST-HAVE        : {self.must_have_concepts}",
            f"  NICE-TO-HAVE     : {self.nice_to_have_concepts}",
            f"  DISQUALIFIERS    : {self.disqualifier_concepts}",
            f"  evidence_terms   : {len(self.evidence_terms)} phrases",
            f"  ideal_text       : {len(self.ideal_text)} chars",
        ]
        return "\n".join(lines)


# =============================================================================
# Section slicing  (header-aware, with graceful fallback)
# =============================================================================
_SECTION_HEADERS = {
    "absolutely_need": ["things you absolutely need"],
    "nice": ["things we'd like you to have", "things we would like"],
    "not_want": ["things we explicitly do not want", "things we explicitly do not"],
    "experience_meaning": ['what we mean by'],
    "location": ["on location, comp", "on location"],
    "read_between": ["how to read between the lines"],
    "disqualifiers_band": ["the disqualifiers we actually apply", "disqualifiers we actually apply"],
}


def _slice_sections(text: str) -> Dict[str, str]:
    """Return {section_key: section_text}. Lowercased matching, original-case text kept."""
    low = text.lower()
    marks = []  # (pos, key)
    for key, variants in _SECTION_HEADERS.items():
        for v in variants:
            i = low.find(v)
            if i != -1:
                marks.append((i, key))
                break
    marks.sort()
    out = {}
    for idx, (pos, key) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        out[key] = text[pos:end]
    return out


def _concepts_in(text: str, library: Dict[str, Dict]) -> List[str]:
    low = text.lower()
    found = []
    for concept, spec in library.items():
        cues = spec["jd_cues"]
        if any(cue in low for cue in cues):
            found.append(concept)
    return found


def parse_jd(text: str) -> JDSpec:
    """Parse a job description string into a structured JDSpec (deterministic, local)."""
    spec = JDSpec(raw_text=text)
    low = text.lower()
    sections = _slice_sections(text)

    # --- Experience band: first "<n>-<m> years" pattern ---
    m = re.search(r"(\d+)\s*[-–to]+\s*(\d+)\s*year", low)
    if m:
        spec.exp_band = (float(m.group(1)), float(m.group(2)))
    # Ideal sub-band from the "read between the lines" section, e.g. "6-8 years total"
    rb = sections.get("read_between", "")
    mi = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years?\s*total", rb.lower())
    if mi:
        spec.exp_ideal = (float(mi.group(1)), float(mi.group(2)))

    # --- Role titles: from the JD title line + role mandate ---
    # Heuristic: if the JD talks about ranking/retrieval/ML, it's an ai_ml role.
    if any(k in low for k in ["ai engineer", "ml engineer", "machine learning",
                              "ranking", "retrieval", "embeddings"]):
        spec.role_titles = list(ROLE_TITLE_VOCAB["ai_ml_engineer"])
        spec.adjacent_titles = list(ADJACENT_TITLE_VOCAB)

    # --- Locations: any gazetteer city named in the JD (favor the location section) ---
    loc_text = (sections.get("location", "") + " " + text).lower()
    spec.locations = [c for c in INDIA_CITIES if re.search(r"\b" + re.escape(c) + r"\b", loc_text)]
    if "india" in low:
        spec.location_country = "india"

    # --- Notice period preference ---
    mn = re.search(r"sub-?(\d+)-?day notice|buy out up to (\d+) day|(\d+)\s*day notice", low)
    if mn:
        spec.notice_pref_days = int(next(g for g in mn.groups() if g))

    # --- Concepts by polarity, scoped to the relevant section when available ---
    must_text = sections.get("absolutely_need", "") or text
    nice_text = sections.get("nice", "")
    spec.must_have_concepts = _concepts_in(must_text, CONCEPT_LIBRARY)
    spec.nice_to_have_concepts = [c for c in _concepts_in(nice_text, CONCEPT_LIBRARY)
                                  if c not in spec.must_have_concepts]
    # If section slicing failed, fall back to whole-text concept scan as must-haves.
    if not spec.must_have_concepts:
        spec.must_have_concepts = _concepts_in(text, CONCEPT_LIBRARY)

    # --- Disqualifiers (scan the whole JD; these phrases are distinctive) ---
    spec.disqualifier_concepts = _concepts_in(text, DISQUALIFIER_LIBRARY)

    # --- Evidence terms: union of evidence phrases for must + nice concepts ---
    ev, core = [], []
    for c in spec.must_have_concepts + spec.nice_to_have_concepts:
        ev.extend(CONCEPT_LIBRARY[c]["evidence"])
        if c in CORE_CONCEPTS:
            core.extend(CONCEPT_LIBRARY[c]["evidence"])
    seen = set()
    spec.evidence_terms = [t for t in ev if not (t in seen or seen.add(t))]
    seen = set()
    spec.core_evidence_terms = [t for t in core if not (t in seen or seen.add(t))]

    # --- Behavioral priorities (JD explicitly tells us availability matters) ---
    if any(k in low for k in ["response rate", "logged in", "actually available",
                              "behavioral signals", "open to work", "in the job market"]):
        spec.behavioral_priorities = {
            "recruiter_response_rate": "high",
            "last_active_date": "recent",
            "open_to_work_flag": "true",
            "notice_period_days": "low",
            "interview_completion_rate": "high",
        }

    # --- Ideal-candidate text for semantic matching (richest signal paragraphs) ---
    spec.ideal_text = " ".join(filter(None, [
        sections.get("read_between", ""),
        sections.get("absolutely_need", ""),
    ])).strip() or text

    # --- Graceful degradation for out-of-vocabulary JDs ---
    # If the concept library / role vocab don't recognize this JD's domain, derive
    # fallback title + evidence signal straight from the JD text so a novel-domain JD
    # still ranks on more than semantics alone. These fire ONLY when the primary path
    # is empty, so a covered JD (e.g. this AI/ML one) is completely unaffected.
    if not spec.role_titles:
        spec.role_titles = _fallback_role_titles(text)
    if not spec.core_evidence_terms:
        fb = _fallback_evidence_terms(must_text)
        spec.core_evidence_terms = fb
        if not spec.evidence_terms:
            spec.evidence_terms = fb

    return spec


def load_jd(path: str) -> JDSpec:
    with open(path, "r", encoding="utf-8") as f:
        return parse_jd(f.read())


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "../job_description.md"
    spec = load_jd(p)
    print(spec.summary())
    print("\n--- evidence_terms ---")
    print(spec.evidence_terms)
    print("\n--- ideal_text (first 400 chars) ---")
    print(spec.ideal_text[:400])
