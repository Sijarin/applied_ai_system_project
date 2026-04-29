"""
Confidence scoring and aggregate evaluation metrics for recommendations.

confidence_score()         – per-song match quality label
evaluate_recommendations() – aggregate metrics over a top-k list
"""

from typing import Any


CONFIDENCE_THRESHOLDS: dict[str, float] = {
    "high":   0.70,  # ≥70% of theoretical maximum
    "medium": 0.45,  # 45–69%
    # low: < 45%
}


def confidence_score(score: float, max_score: float) -> tuple[float, str]:
    """
    Return (ratio, label) where ratio = score/max_score and label is
    'high', 'medium', or 'low'.
    """
    if max_score <= 0:
        return 0.0, "low"
    ratio = min(score / max_score, 1.0)
    if ratio >= CONFIDENCE_THRESHOLDS["high"]:
        label = "high"
    elif ratio >= CONFIDENCE_THRESHOLDS["medium"]:
        label = "medium"
    else:
        label = "low"
    return round(ratio, 4), label


def evaluate_recommendations(
    results: list[tuple[dict, float, list]],
    max_score: float,
) -> dict[str, Any]:
    """
    Compute aggregate metrics over a recommendation list.

    Returns a dict with:
      avg_confidence_ratio  – mean score / max_score across results
      high_confidence_count – number of 'high' confidence results
      genre_diversity       – number of unique genres in the list
      artist_diversity      – number of unique artists in the list
      explicit_count        – number of explicit songs included
    """
    if not results:
        return {}

    ratios: list[float] = []
    high_count = 0
    genres: set[str] = set()
    artists: set[str] = set()
    explicit = 0

    for song, score, _ in results:
        ratio, label = confidence_score(score, max_score)
        ratios.append(ratio)
        if label == "high":
            high_count += 1
        genres.add(song.get("genre", ""))
        artists.add(song.get("artist", ""))
        if int(song.get("explicit", 0)) == 1:
            explicit += 1

    return {
        "avg_confidence_ratio":  round(sum(ratios) / len(ratios), 4),
        "high_confidence_count": high_count,
        "genre_diversity":       len(genres),
        "artist_diversity":      len(artists),
        "explicit_count":        explicit,
    }
