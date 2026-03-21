"""Duplicate and similarity detection using TF-IDF cosine similarity."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def find_duplicates(bookmarks: list, threshold: float = 0.65) -> list:
    """Find similar/duplicate bookmarks based on content similarity."""
    if len(bookmarks) < 2:
        return []

    contents = [b.get("content", "") for b in bookmarks]
    ids = [b.get("tweet_id", "") for b in bookmarks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(contents)
    except ValueError:
        return []

    sim_matrix = cosine_similarity(tfidf_matrix)

    duplicates = []
    seen = set()
    for i in range(len(sim_matrix)):
        for j in range(i + 1, len(sim_matrix)):
            if sim_matrix[i][j] >= threshold:
                pair_key = tuple(sorted([ids[i], ids[j]]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    duplicates.append({
                        "tweet_a": bookmarks[i],
                        "tweet_b": bookmarks[j],
                        "similarity": round(float(sim_matrix[i][j]), 3)
                    })

    return sorted(duplicates, key=lambda x: x["similarity"], reverse=True)


def get_content_embeddings(contents: list) -> np.ndarray:
    """Get TF-IDF embeddings for a list of content strings."""
    if not contents:
        return np.array([])
    vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    try:
        return vectorizer.fit_transform(contents).toarray()
    except ValueError:
        return np.array([])
