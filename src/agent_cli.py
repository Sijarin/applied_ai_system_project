"""
Interactive agentic CLI for the music recommender.

Run with:
    python -m src.agent_cli

The agent loop:
  1. Accept natural language query
  2. Parse to structured prefs via Claude tool use   [RAG + LLM]
  3. Validate with guardrails
  4. Run recommender (with diversity penalty)
  5. Validate output
  6. Generate AI explanations (all songs, one API call) [RAG + LLM]
  7. Display results with confidence scores           [Reliability]
  8. Claude self-critiques the list                   [Agentic loop]
  9. Log everything to logs/events.jsonl              [Logging]
 10. Repeat
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads .env if present; falls back to real env vars

from src.recommender import (
    load_songs,
    recommend_songs,
    diverse_recommend_songs,
    get_max_score,
    SCORING_MODES,
)
from src.rag import list_known_genres, list_known_moods
from src.guardrails import (
    sanitize_text,
    validate_preferences,
    validate_recommendations,
    GuardrailError,
)
from src.evaluation import confidence_score, evaluate_recommendations
from src.ai_agent import (
    plan_query,
    parse_natural_language,
    generate_all_explanations,
    self_critique,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "agent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def _log_event(event_type: str, data: dict) -> None:
    """Append a structured JSON event to the session event log."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    with open(LOGS_DIR / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ── Display helpers ────────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def _bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _print_recommendations(
    results: list,
    max_score: float,
    prefs: dict,
    explanations: list[str],
) -> None:
    """Print ranked recommendations with confidence scores and AI explanations."""
    metrics = evaluate_recommendations(results, max_score)
    print(
        f"\n  [Metrics]  avg confidence: {metrics['avg_confidence_ratio']:.0%}"
        f"  |  genre diversity: {metrics['genre_diversity']}"
        f"  |  artist diversity: {metrics['artist_diversity']}"
        f"  |  high-confidence: {metrics['high_confidence_count']}/{len(results)}"
    )
    print()

    for rank, ((song, score, reasons), explanation) in enumerate(
        zip(results, explanations), 1
    ):
        ratio, label = confidence_score(score, max_score)
        print(f"  #{rank}  {song['title']} — {song['artist']}")
        print(
            f"       Genre: {song['genre']}  |  Mood: {song['mood']}"
            f"  |  BPM: {song['tempo_bpm']}"
        )
        print(
            f"       Score: {score:.2f}/{max_score:.2f}  {_bar(ratio)}"
            f"  [{label} confidence]"
        )
        print(f"       Why: {'; '.join(reasons)}")
        if explanation:
            print(f"       AI:  {explanation}")
        print()


# ── Agent turn ─────────────────────────────────────────────────────────────────

def run_agent_turn(
    query: str,
    songs: list,
    client: Groq,
    mode: str = "balanced",
    use_diversity: bool = True,
) -> None:
    """
    One complete turn of the agentic loop:
    parse → validate → recommend → validate → explain → display → critique.
    """
    # Step 0: Reasoning plan (observable intermediate step)
    print("\n  [Step 0 — Reasoning Plan]")
    try:
        plan = plan_query(query, client)
        signals = ", ".join(plan.get("detected_signals", [])) or "(none detected)"
        print(f"  Signals    : {signals}")
        print(f"  Mode hint  : {plan.get('recommended_mode', 'balanced')}")
        print(f"  Expectation: {plan.get('catalog_expectation', '')}")
        conflicts = plan.get("potential_conflicts", [])
        if conflicts:
            print(f"  Conflicts  : {'; '.join(conflicts)}")
        _log_event("plan", {"plan": plan})
    except Exception as exc:
        log.warning("Planning step failed: %s", exc)
        print(f"  (Plan unavailable: {exc})")
        plan = {}

    # Step 1: Parse natural language → structured prefs
    log.info("Parsing query: %r", query)
    _log_event("query", {"text": query, "mode": mode})

    try:
        raw_prefs = parse_natural_language(query, client)
        log.info("Parsed prefs: %s", raw_prefs)
    except Exception as exc:
        log.error("Parse failed: %s", exc)
        print(f"  [Error] Could not parse your request: {exc}")
        _log_event("parse_error", {"error": str(exc)})
        return

    # Step 2: Input guardrails
    try:
        prefs = validate_preferences(raw_prefs)
        log.info("Validated prefs: %s", prefs)
    except GuardrailError as exc:
        log.warning("Input guardrail blocked: %s", exc)
        print(f"  [Guardrail] {exc}")
        _log_event("guardrail_block", {"error": str(exc)})
        return

    # Step 3: Recommend
    max_score = get_max_score(mode)
    if use_diversity:
        results = diverse_recommend_songs(prefs, songs, k=5, mode=mode)
    else:
        results = recommend_songs(prefs, songs, k=5, mode=mode)

    # Step 4: Output guardrails
    try:
        validate_recommendations(results)
    except GuardrailError as exc:
        log.error("Output guardrail failed: %s", exc)
        print(f"  [Output Error] {exc}")
        return

    _log_event(
        "recommendations",
        {
            "prefs": prefs,
            "mode": mode,
            "diversity": use_diversity,
            "results": [
                {"title": s["title"], "artist": s["artist"], "score": sc}
                for s, sc, _ in results
            ],
        },
    )

    # Step 5: Generate all AI explanations in one API call
    print("\n  Generating AI explanations...")
    try:
        explanations = generate_all_explanations(results, prefs, max_score, client)
    except Exception as exc:
        log.warning("Explanation generation failed: %s", exc)
        explanations = [""] * len(results)

    # Step 6: Display results
    _print_header(f"Recommendations — {mode.upper()} mode")
    _print_recommendations(results, max_score, prefs, explanations)

    # Step 7: Self-critique (agentic self-check)
    print("  [AI Self-Critique]")
    try:
        critique = self_critique(results, prefs, max_score, client)
        print(f"  {critique}\n")
        _log_event("critique", {"critique": critique})
    except Exception as exc:
        log.warning("Self-critique failed: %s", exc)
        print(f"  (Critique unavailable: {exc})\n")


# ── Main interactive loop ──────────────────────────────────────────────────────

def main() -> None:
    _print_header("VibeFinder AI — Agentic Music Recommender")
    print(f"  Known genres : {', '.join(list_known_genres())}")
    print(f"  Known moods  : {', '.join(list_known_moods())}")
    print(f"  Scoring modes: {', '.join(SCORING_MODES.keys())}")
    print()
    print("  Commands:")
    print("    quit               – exit")
    print("    mode <name>        – change scoring mode")
    print("    diversity on/off   – toggle diversity penalty")
    print("    Just type what you want to listen to in plain English!")
    print()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("  [Error] Set the GROQ_API_KEY environment variable and re-run.")
        sys.exit(1)

    client = Groq(api_key=api_key)
    songs = load_songs("data/songs.csv")
    print(f"  Catalog loaded: {len(songs)} songs\n")

    mode = "balanced"
    use_diversity = True

    while True:
        try:
            query = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not query:
            continue

        # Built-in commands
        if query.lower() == "quit":
            print("  Goodbye!")
            break

        if query.lower().startswith("mode "):
            new_mode = query[5:].strip()
            if new_mode in SCORING_MODES:
                mode = new_mode
                print(f"  Mode set to: {mode}")
            else:
                print(
                    f"  Unknown mode '{new_mode}'. "
                    f"Choose from: {', '.join(SCORING_MODES.keys())}"
                )
            continue

        if query.lower() == "diversity off":
            use_diversity = False
            print("  Diversity penalty: OFF")
            continue

        if query.lower() == "diversity on":
            use_diversity = True
            print("  Diversity penalty: ON")
            continue

        # Sanitize input (guardrail)
        try:
            sanitized = sanitize_text(query)
        except GuardrailError as exc:
            print(f"  [Input Guardrail] {exc}")
            continue

        run_agent_turn(
            sanitized, songs, client, mode=mode, use_diversity=use_diversity
        )


if __name__ == "__main__":
    main()
