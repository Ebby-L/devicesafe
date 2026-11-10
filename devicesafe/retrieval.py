"""Lightweight retrieval over adverse event narratives (TF-IDF, no GPU or vector DB)."""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .openfda import Event


class NarrativeIndex:
    def __init__(self, events: list[Event]):
        self.events = [e for e in events if e.narrative]
        if not self.events:
            raise ValueError("No events with narrative text to index.")
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   min_df=1, max_df=0.9, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(e.narrative for e in self.events)

    def search(self, query: str, k: int = 8) -> list[tuple[Event, float]]:
        scores = cosine_similarity(self.vec.transform([query]), self.matrix).ravel()
        top = np.argsort(scores)[::-1][:k]
        return [(self.events[i], float(scores[i])) for i in top if scores[i] > 0]


def problem_frequencies(events: list[Event], top: int = 10) -> list[tuple[str, int]]:
    return Counter(p for e in events for p in e.product_problems).most_common(top)


def event_type_counts(events: list[Event]) -> dict[str, int]:
    return dict(Counter(e.event_type for e in events).most_common())
