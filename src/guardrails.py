"""
Input/output validation guardrails for the music recommender.

All public functions raise GuardrailError on invalid input so callers can
catch a single exception type and present a clean error message.
"""

from typing import Any

VALID_GENRES: frozenset[str] = frozenset({
    "pop", "lofi", "rock", "metal", "hip-hop", "edm", "jazz",
    "ambient", "synthwave", "r&b", "soul", "classical", "country",
    "reggae", "indie pop", "",
})

VALID_MOODS: frozenset[str] = frozenset({
    "happy", "chill", "intense", "focused", "euphoric", "moody",
    "romantic", "melancholic", "angry", "uplifting", "peaceful",
    "nostalgic", "relaxed", "",
})

VALID_MODES: frozenset[str] = frozenset({
    "balanced", "genre_first", "mood_first", "energy_focused", "discovery",
})

MAX_QUERY_LENGTH = 500


class GuardrailError(ValueError):
    """Raised when input or output fails a guardrail check."""


def sanitize_text(text: str) -> str:
    """Strip and validate user-supplied free text."""
    if not isinstance(text, str):
        raise GuardrailError("Input must be a string.")
    text = text.strip()
    if len(text) == 0:
        raise GuardrailError("Input text is empty.")
    if len(text) > MAX_QUERY_LENGTH:
        raise GuardrailError(
            f"Input too long ({len(text)} chars; max {MAX_QUERY_LENGTH})."
        )
    return text


def validate_preferences(prefs: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and coerce a preferences dict.

    Returns a cleaned copy; raises GuardrailError on invalid values.
    Unknown keys are silently dropped so LLM-generated dicts with extra
    fields don't break downstream code.
    """
    cleaned: dict[str, Any] = {}

    # energy: float in [0, 1]
    try:
        energy = float(prefs.get("energy", 0.5))
    except (TypeError, ValueError):
        raise GuardrailError("'energy' must be a number.")
    if not 0.0 <= energy <= 1.0:
        raise GuardrailError(f"'energy' must be 0.0–1.0, got {energy}.")
    cleaned["energy"] = round(energy, 3)

    # genre: must be known or empty
    genre = str(prefs.get("genre", "")).lower().strip()
    if genre not in VALID_GENRES:
        # Attempt partial match (e.g. "hiphop" → "hip-hop")
        normalised = genre.replace(" ", "-").replace("_", "-")
        if normalised in VALID_GENRES:
            genre = normalised
        else:
            raise GuardrailError(
                f"Unknown genre '{genre}'. Known: {sorted(VALID_GENRES - {''})}."
            )
    cleaned["genre"] = genre

    # mood: must be known or empty
    mood = str(prefs.get("mood", "")).lower().strip()
    if mood not in VALID_MOODS:
        raise GuardrailError(
            f"Unknown mood '{mood}'. Known: {sorted(VALID_MOODS - {''})}."
        )
    cleaned["mood"] = mood

    # likes_acoustic: bool
    cleaned["likes_acoustic"] = bool(prefs.get("likes_acoustic", False))

    # popularity: int 0–100
    try:
        popularity = int(prefs.get("popularity", 50))
    except (TypeError, ValueError):
        raise GuardrailError("'popularity' must be an integer.")
    if not 0 <= popularity <= 100:
        raise GuardrailError(f"'popularity' must be 0–100, got {popularity}.")
    cleaned["popularity"] = popularity

    # decade: int, must be 0 or a round-decade year (1950–2030)
    try:
        decade = int(prefs.get("decade", 0))
    except (TypeError, ValueError):
        raise GuardrailError("'decade' must be an integer.")
    if decade != 0 and (decade < 1950 or decade > 2030 or decade % 10 != 0):
        raise GuardrailError(
            f"'decade' must be 0 (any) or a round decade 1950–2030, got {decade}."
        )
    cleaned["decade"] = decade

    # mood_tags: list of strings
    tags = prefs.get("mood_tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    cleaned["mood_tags"] = [str(t).strip().lower() for t in tags]

    # allow_explicit: bool
    cleaned["allow_explicit"] = bool(prefs.get("allow_explicit", True))

    # subgenre: free-form string, cap length
    subgenre = str(prefs.get("subgenre", "")).strip()[:50]
    cleaned["subgenre"] = subgenre

    return cleaned


def validate_recommendations(results: list) -> list:
    """
    Sanity-check a list of (song, score, reasons) tuples.

    Raises GuardrailError if obvious invariants are violated.
    """
    if not isinstance(results, list):
        raise GuardrailError("Recommendations must be a list.")
    if len(results) == 0:
        raise GuardrailError("Recommendations list is empty.")
    for i, item in enumerate(results):
        if not (isinstance(item, tuple) and len(item) == 3):
            raise GuardrailError(f"Item {i} is not a (song, score, reasons) tuple.")
        _, score, reasons = item
        if not isinstance(score, (int, float)):
            raise GuardrailError(f"Item {i} score is not numeric.")
        if not isinstance(reasons, list):
            raise GuardrailError(f"Item {i} reasons is not a list.")
    return results
