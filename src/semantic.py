"""
Semantic matching module.

Uses sentence embeddings to compute semantic similarity between
candidate profiles and the job description.

This catches candidates whose descriptions match the JD semantically,
even if they don't use the exact keywords.

Pre-computation:
    Run `python semantic.py --precompute` to generate embeddings for all candidates.
    This can take several minutes but only needs to be done once.

During ranking:
    Embeddings are loaded from disk and similarity is computed in milliseconds.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from config import SEMANTIC_CONFIG, JD_SUMMARY
from data_loader import extract_text_for_embedding, load_candidates

# Global variables for caching
_model = None
_embeddings = None
_candidate_ids = None


def get_model():
    """
    Get or load the sentence transformer model.

    Uses lazy loading to avoid loading the model if not needed.
    """
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_name = SEMANTIC_CONFIG['model_name']
            print(f"Loading embedding model: {model_name}...")
            _model = SentenceTransformer(model_name)
            print("Model loaded successfully!")
        except ImportError:
            print("WARNING: sentence-transformers not installed.")
            print("Install with: pip install sentence-transformers")
            return None
    return _model


def compute_embeddings(texts: List[str], batch_size: int = 64) -> Optional[np.ndarray]:
    """
    Compute embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding

    Returns:
        numpy array of embeddings (n_texts, embedding_dim)
    """
    model = get_model()
    if model is None:
        return None

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings


def precompute_candidate_embeddings(
    candidates: List[Dict[str, Any]],
    output_dir: str = "."
) -> Tuple[np.ndarray, List[str]]:
    """
    Pre-compute embeddings for all candidates.

    This is done OFFLINE before the ranking step.
    Can take several minutes for 100K candidates.

    Args:
        candidates: List of candidate dictionaries
        output_dir: Directory to save embeddings

    Returns:
        (embeddings array, list of candidate_ids)
    """
    print(f"Pre-computing embeddings for {len(candidates)} candidates...")

    # Extract text for each candidate
    texts = [extract_text_for_embedding(c) for c in candidates]
    candidate_ids = [c['candidate_id'] for c in candidates]

    # Compute embeddings
    embeddings = compute_embeddings(texts, batch_size=SEMANTIC_CONFIG['batch_size'])

    if embeddings is None:
        print("Failed to compute embeddings (model not available)")
        return None, []

    # Save to disk
    output_path = Path(output_dir)
    np.save(output_path / SEMANTIC_CONFIG['embeddings_file'], embeddings)

    with open(output_path / 'candidate_ids.json', 'w') as f:
        json.dump(candidate_ids, f)

    print(f"Saved embeddings to {output_path / SEMANTIC_CONFIG['embeddings_file']}")
    print(f"Embedding shape: {embeddings.shape}")

    return embeddings, candidate_ids


def load_precomputed_embeddings(input_dir: str = ".") -> Tuple[Optional[np.ndarray], List[str]]:
    """
    Load pre-computed embeddings from disk.

    Returns:
        (embeddings array, list of candidate_ids) or (None, []) if not found
    """
    global _embeddings, _candidate_ids

    if _embeddings is not None:
        return _embeddings, _candidate_ids

    input_path = Path(input_dir)
    embeddings_file = input_path / SEMANTIC_CONFIG['embeddings_file']
    ids_file = input_path / 'candidate_ids.json'

    if not embeddings_file.exists():
        print(f"Embeddings file not found: {embeddings_file}")
        return None, []

    if not ids_file.exists():
        print(f"Candidate IDs file not found: {ids_file}")
        return None, []

    print("Loading pre-computed embeddings...")
    _embeddings = np.load(embeddings_file)

    with open(ids_file, 'r') as f:
        _candidate_ids = json.load(f)

    print(f"Loaded {len(_candidate_ids)} embeddings")
    return _embeddings, _candidate_ids


def compute_jd_embedding() -> Optional[np.ndarray]:
    """
    Compute embedding for the job description.

    Returns:
        JD embedding vector (1, embedding_dim)
    """
    model = get_model()
    if model is None:
        return None

    jd_embedding = model.encode([JD_SUMMARY], convert_to_numpy=True)

    # Normalize
    norm = np.linalg.norm(jd_embedding)
    jd_embedding = jd_embedding / (norm + 1e-8)

    return jd_embedding


def compute_semantic_scores(
    candidates: List[Dict[str, Any]],
    embeddings_dir: str = "."
) -> Dict[str, float]:
    """
    Compute semantic similarity scores for all candidates.

    If pre-computed embeddings exist, use them.
    Otherwise, compute on-the-fly (slower).

    Args:
        candidates: List of candidate dictionaries
        embeddings_dir: Directory containing pre-computed embeddings

    Returns:
        Dictionary mapping candidate_id to semantic score (0-1)
    """
    scores = {}

    # Try to load pre-computed embeddings
    embeddings, candidate_ids = load_precomputed_embeddings(embeddings_dir)

    if embeddings is not None:
        # Use pre-computed embeddings
        jd_embedding = compute_jd_embedding()
        if jd_embedding is None:
            return {c['candidate_id']: 0.5 for c in candidates}  # Default score

        # Compute cosine similarities (dot product since normalized)
        similarities = np.dot(embeddings, jd_embedding.T).flatten()

        # Normalize to 0-1 range (similarities are typically -1 to 1)
        similarities = (similarities + 1) / 2

        # Create mapping
        id_to_score = dict(zip(candidate_ids, similarities))
        scores = {c['candidate_id']: id_to_score.get(c['candidate_id'], 0.5)
                  for c in candidates}
    else:
        # Compute on-the-fly (slower but works without pre-computation)
        model = get_model()
        if model is None:
            # No model available, return neutral scores
            return {c['candidate_id']: 0.5 for c in candidates}

        jd_embedding = compute_jd_embedding()
        if jd_embedding is None:
            return {c['candidate_id']: 0.5 for c in candidates}

        # Compute in batches
        batch_size = 100
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i+batch_size]
            texts = [extract_text_for_embedding(c) for c in batch]
            batch_embeddings = compute_embeddings(texts, batch_size=len(texts))

            if batch_embeddings is not None:
                similarities = np.dot(batch_embeddings, jd_embedding.T).flatten()
                similarities = (similarities + 1) / 2  # Normalize to 0-1

                for c, sim in zip(batch, similarities):
                    scores[c['candidate_id']] = float(sim)

    return scores


def get_semantic_score(candidate_id: str, semantic_scores: Dict[str, float]) -> float:
    """
    Get semantic score for a single candidate.

    Args:
        candidate_id: The candidate's ID
        semantic_scores: Pre-computed scores dictionary

    Returns:
        Semantic similarity score (0-1)
    """
    return semantic_scores.get(candidate_id, 0.5)  # Default to neutral


# CLI for pre-computation
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Semantic embedding utilities')
    parser.add_argument('--precompute', action='store_true',
                        help='Pre-compute embeddings for all candidates')
    parser.add_argument('--candidates', type=str, default='../candidates.jsonl',
                        help='Path to candidates file')
    parser.add_argument('--output', type=str, default='.',
                        help='Output directory for embeddings')

    args = parser.parse_args()

    if args.precompute:
        candidates = load_candidates(args.candidates)
        precompute_candidate_embeddings(candidates, args.output)
        print("Pre-computation complete!")
