"""
Offline evaluation harness — no Groq API key required.

Runs predefined test cases through the deterministic pipeline layers:
  guardrails · recommender · evaluator · RAG retriever

Usage:
    python tests/test_harness.py        (from project root)
    python -m tests.test_harness        (alternative)

Exit code: 0 if all pass, 1 if any fail.
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.guardrails import sanitize_text, validate_preferences, validate_recommendations, GuardrailError
from src.recommender import load_songs, recommend_songs, diverse_recommend_songs, get_max_score
from src.evaluation import confidence_score, evaluate_recommendations
from src.rag import retrieve_context, list_known_decades

SONGS = load_songs(str(Path(__file__).parent.parent / "data" / "songs.csv"))


# ── Test case dataclass ────────────────────────────────────────────────────────

@dataclass
class Case:
    name: str
    layer: str
    fn: Callable[[], tuple[bool, str]]   # returns (passed, detail)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prefs(**overrides) -> dict:
    base = {
        "genre": "", "mood": "", "energy": 0.5, "likes_acoustic": False,
        "popularity": 50, "decade": 0, "mood_tags": [], "allow_explicit": True, "subgenre": "",
    }
    base.update(overrides)
    return base


# ── Test functions ─────────────────────────────────────────────────────────────

def tc_sanitize_strips_whitespace():
    result = sanitize_text("  chill lo-fi beats  ")
    ok = result == "chill lo-fi beats"
    return ok, f"'{result}'"


def tc_sanitize_rejects_empty():
    try:
        sanitize_text("   ")
        return False, "did not raise GuardrailError"
    except GuardrailError:
        return True, "GuardrailError raised correctly"


def tc_sanitize_rejects_too_long():
    try:
        sanitize_text("x" * 501)
        return False, "did not raise GuardrailError"
    except GuardrailError:
        return True, "GuardrailError raised for 501-char input"


def tc_validate_prefs_valid():
    prefs = validate_preferences(_prefs(genre="pop", mood="happy", energy=0.8))
    ok = prefs["genre"] == "pop" and prefs["mood"] == "happy"
    return ok, f"genre={prefs['genre']}, mood={prefs['mood']}"


def tc_validate_prefs_bad_energy():
    try:
        validate_preferences(_prefs(energy=1.5))
        return False, "did not raise GuardrailError"
    except GuardrailError:
        return True, "GuardrailError raised for energy=1.5"


def tc_validate_prefs_bad_genre():
    try:
        validate_preferences(_prefs(genre="death_polka"))
        return False, "did not raise GuardrailError"
    except GuardrailError:
        return True, "GuardrailError raised for genre='death_polka'"


def tc_recommender_top1_genre():
    prefs = _prefs(genre="lofi", mood="chill", energy=0.38, likes_acoustic=True)
    results = recommend_songs(prefs, SONGS, k=5)
    top_genre = results[0][0]["genre"]
    ok = top_genre == "lofi"
    score = round(results[0][1], 2)
    return ok, f"top result genre='{top_genre}', score={score}"


def tc_recommender_confidence_high():
    prefs = _prefs(genre="pop", mood="happy", energy=0.82, likes_acoustic=False,
                   popularity=85, decade=2020)
    results = recommend_songs(prefs, SONGS, k=5, mode="balanced")
    max_score = get_max_score("balanced")
    ratio, label = confidence_score(results[0][1], max_score)
    ok = label == "high"
    return ok, f"label={label} ({ratio:.0%} of max_score={max_score})"


def tc_recommender_diverse_unique_artists():
    prefs = _prefs(genre="lofi", mood="chill", energy=0.38, likes_acoustic=True)
    results = diverse_recommend_songs(prefs, SONGS, k=5)
    artists = [r[0]["artist"] for r in results]
    unique = len(set(artists))
    ok = unique >= 4
    return ok, f"{unique}/{len(artists)} unique artists"


def tc_explicit_penalized_when_blocked():
    high_energy = _prefs(energy=0.95, likes_acoustic=False, popularity=75)
    allow_prefs  = {**high_energy, "allow_explicit": True}
    block_prefs  = {**high_energy, "allow_explicit": False}

    all_songs_allow = recommend_songs(allow_prefs, SONGS, k=len(SONGS))
    all_songs_block = recommend_songs(block_prefs, SONGS, k=len(SONGS))

    explicit_songs = [s for s, _, _ in all_songs_allow if int(s.get("explicit", 0)) == 1]
    if not explicit_songs:
        return True, "no explicit songs in catalog — skipped"

    target = explicit_songs[0]["title"]
    rank_allow = next(i for i, (s, _, _) in enumerate(all_songs_allow) if s["title"] == target)
    rank_block = next(i for i, (s, _, _) in enumerate(all_songs_block) if s["title"] == target)
    ok = rank_block > rank_allow
    return ok, f"'{target}' rank: #{rank_allow+1} → #{rank_block+1} when explicit blocked"


def tc_evaluation_aggregate_metrics():
    prefs = _prefs(genre="rock", mood="intense", energy=0.91)
    results = recommend_songs(prefs, SONGS, k=5)
    max_score = get_max_score("balanced")
    metrics = evaluate_recommendations(results, max_score)
    required_keys = {"avg_confidence_ratio", "genre_diversity", "artist_diversity", "high_confidence_count"}
    ok = required_keys.issubset(metrics.keys())
    avg = metrics.get("avg_confidence_ratio", 0)
    gd  = metrics.get("genre_diversity", 0)
    ad  = metrics.get("artist_diversity", 0)
    return ok, f"avg_conf={avg:.0%}, genre_div={gd}, artist_div={ad}"


def tc_rag_genre_and_mood():
    ctx = retrieve_context(genre="lofi", mood="chill")
    ok = "[Genre: lofi]" in ctx and "[Mood: chill]" in ctx
    return ok, f"{len(ctx)} chars returned, both sections present"


def tc_rag_decade_third_source():
    ctx = retrieve_context(genre="synthwave", decade=1980)
    ok = "[Genre: synthwave]" in ctx and "[Decade: 1980s]" in ctx
    return ok, f"genre + era context combined ({len(ctx)} chars)"


def tc_rag_all_three_sources():
    ctx = retrieve_context(genre="pop", mood="happy", decade=2020)
    ok = all(tag in ctx for tag in ["[Genre: pop]", "[Mood: happy]", "[Decade: 2020s]"])
    return ok, f"all three data sources present ({len(ctx)} chars)"


# ── Case registry ──────────────────────────────────────────────────────────────

CASES: list[Case] = [
    Case("sanitize:strips_whitespace",   "guardrails",   tc_sanitize_strips_whitespace),
    Case("sanitize:rejects_empty",       "guardrails",   tc_sanitize_rejects_empty),
    Case("sanitize:rejects_too_long",    "guardrails",   tc_sanitize_rejects_too_long),
    Case("prefs:valid_full_dict",        "guardrails",   tc_validate_prefs_valid),
    Case("prefs:bad_energy_range",       "guardrails",   tc_validate_prefs_bad_energy),
    Case("prefs:unknown_genre",          "guardrails",   tc_validate_prefs_bad_genre),
    Case("recommend:top1_genre_match",   "recommender",  tc_recommender_top1_genre),
    Case("recommend:high_confidence",    "recommender",  tc_recommender_confidence_high),
    Case("recommend:diverse_artists",    "recommender",  tc_recommender_diverse_unique_artists),
    Case("recommend:explicit_penalized", "recommender",  tc_explicit_penalized_when_blocked),
    Case("eval:aggregate_metrics",       "evaluator",    tc_evaluation_aggregate_metrics),
    Case("rag:genre_and_mood",           "rag",          tc_rag_genre_and_mood),
    Case("rag:decade_third_source",      "rag",          tc_rag_decade_third_source),
    Case("rag:all_three_sources",        "rag",          tc_rag_all_three_sources),
]


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all() -> bool:
    name_w  = max(len(c.name)  for c in CASES) + 2
    layer_w = max(len(c.layer) for c in CASES) + 2

    print(f"\n{'═' * 78}")
    print("  VibeFinder AI — Evaluation Harness  (no Groq API key required)")
    print(f"{'═' * 78}")
    print(f"  {'Test Case':<{name_w}}  {'Layer':<{layer_w}}  {'Result':<8}  Detail")
    print(f"  {'─' * name_w}  {'─' * layer_w}  {'─' * 8}  {'─' * 35}")

    passed = 0
    failed = 0

    for case in CASES:
        try:
            ok, detail = case.fn()
        except Exception as exc:
            ok = False
            detail = f"EXCEPTION: {exc}"

        marker = "✓" if ok else "✗"
        status = "PASS" if ok else "FAIL"
        print(f"  {case.name:<{name_w}}  {case.layer:<{layer_w}}  {marker} {status:<6}  {detail}")

        if ok:
            passed += 1
        else:
            failed += 1

    total = passed + failed
    print(f"\n{'═' * 78}")
    if failed == 0:
        print(f"  RESULT: {passed}/{total} passed — all checks passed ✓")
    else:
        print(f"  RESULT: {passed}/{total} passed — {failed} FAILED ✗")
    print(f"{'═' * 78}\n")

    return failed == 0


if __name__ == "__main__":
    all_passed = run_all()
    sys.exit(0 if all_passed else 1)
