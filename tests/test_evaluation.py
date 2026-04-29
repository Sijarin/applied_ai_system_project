from src.evaluation import confidence_score, evaluate_recommendations


# ── confidence_score ───────────────────────────────────────────────────────────

def test_perfect_score_is_high():
    ratio, label = confidence_score(5.0, 5.0)
    assert ratio == 1.0
    assert label == "high"


def test_zero_score_is_low():
    ratio, label = confidence_score(0.0, 5.0)
    assert label == "low"


def test_medium_boundary():
    # 0.45 * max_score = medium threshold
    ratio, label = confidence_score(2.25, 5.0)
    assert label == "medium"


def test_high_boundary():
    ratio, label = confidence_score(3.5, 5.0)
    assert label == "high"


def test_zero_max_score_returns_low():
    ratio, label = confidence_score(1.0, 0.0)
    assert ratio == 0.0
    assert label == "low"


def test_ratio_capped_at_one():
    ratio, label = confidence_score(10.0, 5.0)
    assert ratio == 1.0


# ── evaluate_recommendations ───────────────────────────────────────────────────

def _song(genre="pop", artist="ArtistA", explicit=0):
    return {
        "title": "T",
        "artist": artist,
        "genre": genre,
        "explicit": explicit,
    }


def test_empty_results_returns_empty():
    assert evaluate_recommendations([], 5.0) == {}


def test_single_high_confidence_result():
    results = [(_song(), 4.0, [])]
    metrics = evaluate_recommendations(results, 5.0)
    assert metrics["high_confidence_count"] == 1
    assert metrics["genre_diversity"] == 1
    assert metrics["artist_diversity"] == 1
    assert metrics["explicit_count"] == 0


def test_explicit_counted():
    results = [(_song(explicit=1), 3.0, []), (_song(explicit=0), 3.0, [])]
    metrics = evaluate_recommendations(results, 5.0)
    assert metrics["explicit_count"] == 1


def test_artist_diversity():
    results = [
        (_song(artist="ArtistA"), 3.0, []),
        (_song(artist="ArtistB"), 3.0, []),
        (_song(artist="ArtistA"), 3.0, []),  # repeat
    ]
    metrics = evaluate_recommendations(results, 5.0)
    assert metrics["artist_diversity"] == 2


def test_genre_diversity():
    results = [
        (_song(genre="pop"), 3.0, []),
        (_song(genre="rock"), 3.0, []),
        (_song(genre="lofi"), 3.0, []),
    ]
    metrics = evaluate_recommendations(results, 5.0)
    assert metrics["genre_diversity"] == 3


def test_avg_confidence_ratio():
    results = [(_song(), 5.0, []), (_song(), 0.0, [])]
    metrics = evaluate_recommendations(results, 5.0)
    assert metrics["avg_confidence_ratio"] == 0.5
