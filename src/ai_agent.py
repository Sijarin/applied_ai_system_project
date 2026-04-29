"""
Groq-powered AI agent layer for the music recommender.

Four public functions (each = one API call):
  plan_query()                    – free text  → structured reasoning plan (function calling)
  parse_natural_language()        – free text  → structured prefs (function calling)
  generate_all_explanations()     – top-k results → list of prose explanations (few-shot)
  self_critique()                 – recommendation list → quality assessment
"""

import json
from typing import Any

from groq import Groq

from src.rag import retrieve_context
from src.evaluation import confidence_score

MODEL = "llama-3.3-70b-versatile"

# ── Tool schema and system prompt for query planning ──────────────────────────

_PLAN_TOOL = {
    "type": "function",
    "function": {
        "name": "create_query_plan",
        "description": (
            "Analyse a music request and produce a structured reasoning plan "
            "before any parsing or recommendation happens."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "detected_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Signals found in the query: genre cues, mood cues, "
                        "activity context, era/decade hints, energy cues, explicit-content preference."
                    ),
                },
                "recommended_mode": {
                    "type": "string",
                    "enum": ["balanced", "genre_first", "mood_first", "energy_focused", "discovery"],
                    "description": "The scoring mode best suited to this query.",
                },
                "catalog_expectation": {
                    "type": "string",
                    "description": "Brief prediction of how many catalog matches to expect and why.",
                },
                "potential_conflicts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Contradictions or ambiguities in the query that may be hard to satisfy.",
                },
            },
            "required": [
                "detected_signals", "recommended_mode",
                "catalog_expectation", "potential_conflicts",
            ],
        },
    },
}

_PLAN_SYSTEM = (
    "You are a music AI planner. Before parsing a user query into preferences, "
    "analyse what signals are present and decide how to approach the recommendation. "
    "Call create_query_plan with your structured reasoning. "
    "Be honest about conflicts or ambiguities — do not paper over them."
)


def plan_query(query: str, client: Groq) -> dict[str, Any]:
    """
    Use Groq function calling to produce a structured reasoning plan.

    Returns a dict with keys:
        detected_signals   : list[str]
        recommended_mode   : str  (one of balanced/genre_first/mood_first/energy_focused/discovery)
        catalog_expectation: str
        potential_conflicts: list[str]
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": query},
        ],
        tools=[_PLAN_TOOL],
        tool_choice={"type": "function", "function": {"name": "create_query_plan"}},
        max_tokens=512,
    )
    message = response.choices[0].message
    if message.tool_calls:
        return json.loads(message.tool_calls[0].function.arguments)
    raise RuntimeError("Groq did not return a function call for plan_query.")


# ── Tool schema for preference parsing ────────────────────────────────────────
_PARSE_TOOL = {
    "type": "function",
    "function": {
        "name": "set_music_preferences",
        "description": (
            "Extract structured music preferences from the user's natural language request. "
            "Use empty string for fields the user has not specified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "Music genre (e.g. 'pop', 'lofi', 'rock', 'edm', 'jazz'). Empty string if unspecified.",
                },
                "mood": {
                    "type": "string",
                    "description": "Emotional mood (e.g. 'happy', 'chill', 'intense', 'melancholic'). Empty string if unspecified.",
                },
                "energy": {
                    "type": "number",
                    "description": "Energy level 0.0 (very calm) to 1.0 (very intense). Default 0.5.",
                },
                "likes_acoustic": {
                    "type": "boolean",
                    "description": "True if user wants acoustic/organic sound; False for electronic.",
                },
                "popularity": {
                    "type": "integer",
                    "description": "Preferred mainstream-ness 0-100. Default 50.",
                },
                "decade": {
                    "type": "integer",
                    "description": "Preferred release decade as a round number (1990, 2000, 2010, 2020). 0 = no era preference.",
                },
                "mood_tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Detailed mood descriptors (e.g. ['euphoric', 'danceable']).",
                },
                "allow_explicit": {
                    "type": "boolean",
                    "description": "False if user wants clean/family-friendly content only.",
                },
                "subgenre": {
                    "type": "string",
                    "description": "Specific subgenre (e.g. 'lo-fi hip hop', 'alternative rock'). Empty string if unspecified.",
                },
            },
            "required": [
                "genre", "mood", "energy", "likes_acoustic", "popularity",
                "decade", "mood_tags", "allow_explicit", "subgenre",
            ],
        },
    },
}

_PARSE_SYSTEM = (
    "You are a music preference parser. "
    "When given a user's description of what they want to listen to, "
    "call the set_music_preferences function with the most accurate structured representation. "
    "Be conservative: only set specific values you can confidently infer from the text. "
    "Use empty strings / 0 / 0.5 for unspecified fields."
)


def parse_natural_language(text: str, client: Groq) -> dict[str, Any]:
    """
    Use Groq function calling to extract structured preferences from free text.

    Returns a raw dict (caller should validate with guardrails.validate_preferences).
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _PARSE_SYSTEM},
            {"role": "user", "content": text},
        ],
        tools=[_PARSE_TOOL],
        tool_choice={"type": "function", "function": {"name": "set_music_preferences"}},
        max_tokens=512,
    )
    message = response.choices[0].message
    if message.tool_calls:
        return json.loads(message.tool_calls[0].function.arguments)
    raise RuntimeError("Groq did not return a function call — unexpected response format.")


# ── Bulk explanation generation ────────────────────────────────────────────────

_EXPLAIN_SYSTEM = (
    "You are a music expert who writes vivid, specific song explanations. "
    "Each explanation must be exactly 2 sentences. "
    "Sentence 1 must describe a concrete sonic quality — instrumentation, BPM, texture, or production style. "
    "Sentence 2 must describe the emotional payoff for the listener. "
    "Never repeat the genre label or score. "
    "Avoid filler phrases like 'perfect for', 'great choice', or 'you will love this'.\n\n"
    "Style: warm, opinionated, non-generic. Ground every claim in the music-domain knowledge provided.\n\n"
    "Examples of the required format:\n"
    "EXAMPLE A — pop/happy, energy=0.82, BPM=118:\n"
    "  Sunrise City layers shimmering synth arpeggios over a four-on-the-floor kick at 118 BPM, "
    "giving it an effortlessly danceable momentum. "
    "The high-valence chorus delivers a genuine rush of optimism that suits a morning commute or a "
    "gym warm-up equally well.\n\n"
    "EXAMPLE B — lofi/chill, energy=0.35, BPM=72:\n"
    "  Library Rain pairs a muffled, tape-saturated piano loop with brushed snare hits at 72 BPM, "
    "creating a deliberately imperfect warmth that modern hi-fi recordings rarely achieve. "
    "The nostalgic grain of the recording makes it feel like revisiting a favourite coffee shop "
    "from years ago.\n\n"
    "Now write one 2-sentence explanation per song in the numbered list provided."
)


def generate_all_explanations(
    results: list[tuple[dict, float, list]],
    prefs: dict[str, Any],
    max_score: float,
    client: Groq,
) -> list[str]:
    """
    Generate natural language explanations for every recommendation in one API call.

    Returns a list of strings in the same order as `results`.
    """
    rag_context = retrieve_context(
        prefs.get("genre", ""), prefs.get("mood", ""), int(prefs.get("decade", 0))
    )

    songs_text = ""
    for i, (song, score, reasons) in enumerate(results, 1):
        ratio, conf = confidence_score(score, max_score)
        songs_text += (
            f"\n{i}. '{song['title']}' by {song['artist']}"
            f" (genre: {song['genre']}, mood: {song['mood']},"
            f" energy: {song['energy']}, BPM: {song['tempo_bpm']},"
            f" confidence: {conf} {ratio:.0%})"
            f"\n   Scoring reasons: {'; '.join(reasons)}"
        )

    user_content = (
        f"Music knowledge context:\n{rag_context}\n\n"
        f"User wants: genre={prefs.get('genre') or 'any'}, "
        f"mood={prefs.get('mood') or 'any'}, "
        f"energy={prefs.get('energy', 0.5)}, "
        f"acoustic={prefs.get('likes_acoustic', False)}\n\n"
        f"Songs to explain:{songs_text}\n\n"
        f"Write exactly one 2-sentence explanation per song, numbered 1-{len(results)}. "
        f"Format: '1. <explanation>' on its own line, then '2. <explanation>', etc."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _EXPLAIN_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
    )
    raw = response.choices[0].message.content.strip()

    explanations: list[str] = []
    lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
    for line in lines:
        if len(line) >= 3 and line[0].isdigit() and line[1] == ".":
            explanations.append(line[2:].strip())
        elif len(line) >= 4 and line[:2].isdigit() and line[2] == ".":
            explanations.append(line[3:].strip())

    while len(explanations) < len(results):
        explanations.append("")
    return explanations[: len(results)]


# ── Self-critique ──────────────────────────────────────────────────────────────

_CRITIQUE_SYSTEM = (
    "You are a critical evaluator of music recommendation systems. "
    "Be honest and concise. Identify real issues — diversity, mismatches, "
    "confidence gaps — but also note what the recommendation set gets right. "
    "Keep your evaluation to 3-4 sentences total."
)


def self_critique(
    results: list[tuple[dict, float, list]],
    prefs: dict[str, Any],
    max_score: float,
    client: Groq,
) -> str:
    """
    Ask Groq to evaluate the recommendation list for quality and diversity.

    Returns a brief critique string.
    """
    songs_summary = "\n".join(
        f"  {i+1}. '{s['title']}' by {s['artist']}"
        f" [genre: {s['genre']}, mood: {s['mood']}, score: {sc:.2f}/{max_score:.2f}]"
        for i, (s, sc, _) in enumerate(results)
    )

    user_msg = (
        f"I recommended these songs to a user who wants:\n"
        f"  genre={prefs.get('genre') or 'any'}, "
        f"mood={prefs.get('mood') or 'any'}, "
        f"energy={prefs.get('energy', 0.5)}, "
        f"allow_explicit={prefs.get('allow_explicit', True)}\n\n"
        f"Recommendations:\n{songs_summary}\n\n"
        "Evaluate: Are these recommendations diverse across genre and artist? "
        "Are there obvious mismatches or confidence issues? "
        "What's the overall quality of this set?"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _CRITIQUE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()
