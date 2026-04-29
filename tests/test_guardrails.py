import pytest
from src.guardrails import (
    sanitize_text,
    validate_preferences,
    validate_recommendations,
    GuardrailError,
)


# ── sanitize_text ──────────────────────────────────────────────────────────────

def test_sanitize_strips_whitespace():
    assert sanitize_text("  hello  ") == "hello"


def test_sanitize_rejects_empty():
    with pytest.raises(GuardrailError):
        sanitize_text("   ")


def test_sanitize_rejects_non_string():
    with pytest.raises(GuardrailError):
        sanitize_text(123)


def test_sanitize_rejects_too_long():
    with pytest.raises(GuardrailError):
        sanitize_text("a" * 501)


def test_sanitize_accepts_max_length():
    assert len(sanitize_text("a" * 500)) == 500


# ── validate_preferences ───────────────────────────────────────────────────────

def test_valid_preferences_pass():
    prefs = {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
        "popularity": 75,
        "decade": 2020,
        "mood_tags": ["euphoric"],
        "allow_explicit": True,
        "subgenre": "dance pop",
    }
    result = validate_preferences(prefs)
    assert result["genre"] == "pop"
    assert result["mood"] == "happy"
    assert result["energy"] == 0.8


def test_empty_genre_and_mood_allowed():
    prefs = {"genre": "", "mood": "", "energy": 0.5}
    result = validate_preferences(prefs)
    assert result["genre"] == ""
    assert result["mood"] == ""


def test_energy_out_of_range_raises():
    with pytest.raises(GuardrailError):
        validate_preferences({"energy": 1.5})


def test_unknown_genre_raises():
    with pytest.raises(GuardrailError):
        validate_preferences({"genre": "death_polka"})


def test_unknown_mood_raises():
    with pytest.raises(GuardrailError):
        validate_preferences({"mood": "vibing"})


def test_invalid_decade_raises():
    with pytest.raises(GuardrailError):
        validate_preferences({"decade": 2025})


def test_valid_decade_passes():
    result = validate_preferences({"decade": 2010})
    assert result["decade"] == 2010


def test_zero_decade_means_any():
    result = validate_preferences({"decade": 0})
    assert result["decade"] == 0


def test_popularity_out_of_range_raises():
    with pytest.raises(GuardrailError):
        validate_preferences({"popularity": 101})


def test_mood_tags_as_string_converted():
    result = validate_preferences({"mood_tags": "euphoric,danceable"})
    assert result["mood_tags"] == ["euphoric", "danceable"]


def test_defaults_are_sensible():
    result = validate_preferences({})
    assert result["energy"] == 0.5
    assert result["popularity"] == 50
    assert result["allow_explicit"] is True
    assert result["mood_tags"] == []


# ── validate_recommendations ───────────────────────────────────────────────────

def _make_result(title="Song", score=3.0):
    song = {"title": title, "artist": "Artist", "genre": "pop", "explicit": 0}
    return (song, score, ["genre match (+2.0)"])


def test_valid_recommendations_pass():
    results = [_make_result("A"), _make_result("B")]
    assert validate_recommendations(results) == results


def test_empty_recommendations_raise():
    with pytest.raises(GuardrailError):
        validate_recommendations([])


def test_non_tuple_item_raises():
    with pytest.raises(GuardrailError):
        validate_recommendations([{"title": "X"}])


def test_non_numeric_score_raises():
    song = {"title": "X", "genre": "pop", "explicit": 0}
    with pytest.raises(GuardrailError):
        validate_recommendations([(song, "high", [])])
