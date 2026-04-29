# Model Card: VibeFinder AI — Agentic Music Recommender

## 1. Model Name

**VibeFinder AI 2.0** (extended from the Module 3 Music Recommender Simulator)

---

## 2. Base Project

The original project, **Music Recommender Simulator (Module 3)**, was a content-based
recommender: given a structured taste profile (genre, mood, energy, acoustic preference),
it scored every song in a CSV catalog using a weighted rule system and returned the top-5
matches. This model card covers the full extended system built on top of that foundation.

---

## 3. Goal / Task

VibeFinder AI turns a plain-English music request — *"something chill and acoustic for
studying"* — into a ranked list of songs with explanations. The system handles intent
parsing, preference validation, scoring, confidence labeling, prose generation, and
self-critique entirely automatically. The user only needs to type.

---

## 4. Data Used

The catalog has **18 songs** in `data/songs.csv`. Each song has 10 features: id, title,
artist, genre, mood, energy (0–1), tempo in BPM, valence (0–1), danceability (0–1), and
acousticness (0–1). Genres covered: pop, lofi, rock, ambient, jazz, synthwave, indie pop,
r&b, hip-hop, classical, edm, country, metal, soul, reggae.

The RAG knowledge base (`src/rag.py`) adds three static text sources:
- `GENRE_DOCS` — sonic and production descriptions for each genre
- `MOOD_DOCS` — emotional and contextual descriptions for each mood label
- `DECADE_DOCS` — era-specific production context for 1980s–2020s

**Data limits:**
- Most genres have only one or two catalog songs.
- All numeric features (energy, acousticness, etc.) were assigned by hand, not measured
  from real audio.
- The catalog covers Western popular music from roughly 2000–2025 only.
- The RAG knowledge base is a static Python dict — it cannot be updated without a code
  change.

---

## 5. Algorithm Summary

The pipeline has five stages:

1. **Intent parsing:** Groq LLM (llama-3.3-70b-versatile) reads the user's text and uses
   function calling to emit a structured preferences dict: genre, mood, energy, BPM,
   decade, acoustic preference, explicit content flag, and scoring mode.

2. **Retrieval-augmented context:** `retrieve_context()` looks up the relevant genre, mood,
   and decade paragraphs and injects them into the parser and explainer prompts.

3. **Weighted scoring + MMR:** `recommender.py` scores each song on four signals — genre
   match (+2.0 pts), mood match (+1.0 pt), energy proximity (up to +1.5 pts), acoustic
   match (+0.5 pt) — then applies a greedy re-ranking algorithm that subtracts artist and
   genre repeat penalties to ensure diversity (maximum score: 9.45 in balanced mode with
   all bonuses applied).

4. **Confidence labeling:** `evaluation.py` assigns each result a confidence label (high /
   medium / low) based on its score as a fraction of the theoretical maximum.

5. **Self-critique (agentic loop):** After results are displayed, a fourth Groq LLM call
   evaluates the list for diversity, confidence pattern, and mode alignment, then surfaces
   a plain-English critique to the user.

---

## 6. Observed Biases and Limitations

**Genre dominance.** Genre is the highest-weight signal (2.0 / 9.45 max). A perfect
mood + energy + acoustic match without a genre match will consistently rank below a weak
genre match. This encodes a preference for genre identity over sonic similarity.

**No genre adjacency.** The scoring rule treats every genre mismatch as equally wrong.
Metal and pop both score zero for a rock listener, even though metal is sonically far
closer to rock than pop. This is a structural gap — no weight adjustment fixes a missing
feature.

**Catalog underrepresentation.** Ten genres have exactly one song. Users whose preferences
fall into underserved genres get one strong match and four energy-proximity fallbacks. The
recommendations reflect who the catalog was built for.

**Binary acoustic preference.** The `likes_acoustic` flag is boolean. Real preferences
exist on a spectrum — a user might enjoy both acoustic folk and electronic ambient.

**Neutral users are poorly served.** Without genre or mood preferences, the maximum
achievable confidence is about 32% of the theoretical maximum.

**LLM parsing failures.** If the user's input is ambiguous, the Groq model can parse
"vibing" as a mood, which the guardrail catches and rejects. The system degrades to an
error message rather than a bad recommendation — the right behavior, but it forces a
rephrase.

**Potential misuse — filter bubble reinforcement.** The genre_first and mood_first modes
could be used to build a playlist experience that never exposes the listener to anything
outside their existing preferences. Discovery mode and the diversity penalty counter this
but require the user to opt in.

**Prompt injection via natural language.** User text is passed to an LLM. A crafted input
could attempt to manipulate the parser. The current mitigation is post-LLM validation:
`validate_preferences()` rejects any output that does not pass strict allowlists,
independent of what the LLM produced.

---

## 7. Evaluation Process and Testing Results

### Automated tests — 42/42 pass

```
pytest tests/ -v
42 passed in 0.04s
```

| File | Tests | What is covered |
|---|---|---|
| `test_guardrails.py` | 20 | sanitize_text (empty, too long, non-string), validate_preferences (all 9 fields, boundary values, normalization), validate_recommendations (shape, empty list, bad score type) |
| `test_evaluation.py` | 12 | confidence_score at all thresholds and edge cases, evaluate_recommendations for diversity counts and average confidence |
| `test_rag.py` | 8 | retrieve_context for known genre, known mood, decade, combined, unknown inputs, case-insensitivity |
| `test_recommender.py` | 2 | recommend returns songs sorted descending by score; explain_recommendation returns a non-empty string |

Two real bugs were caught by tests before they reached the LLM:
- `validate_preferences` silently accepted `decade=2025` (not a round decade). Fixed by
  adding a `% 10 != 0` check.
- `mood_tags` passed as a comma-separated string was not being split. Fixed by adding a
  string split branch in `validate_preferences`.

### Evaluation harness — 14/14 pass

```
python tests/test_harness.py
RESULT: 14/14 passed — all checks passed ✓
```

Notable: `recommend:explicit_penalized` confirmed "Iron Curtain" drops from rank #4 to
#17 when `allow_explicit=False`. `recommend:diverse_artists` confirmed the MMR algorithm
produces 5 unique artists from 18 songs.

### Confidence scoring across query types

| Query type | Avg confidence | High-confidence | Notes |
|---|---|---|---|
| Clear genre + mood (chill lo-fi) | 87% | 4/5 | Strong catalog coverage |
| Genre-agnostic, high energy (gym) | 79% | 3/5 | Slots 3–5 are energy-only matches |
| Discovery mode (cross-genre) | 61% | 1/5 | Expected — mode trades confidence for variety |
| Conflicting prefs (high energy + melancholic) | 52% | 1/5 | Energy signal dominates; mood not satisfied |

### Human evaluation — 4 live sessions

| Session | Query intent | Result quality | Self-critique accurate? |
|---|---|---|---|
| 1 | Chill lo-fi for studying | Correct — top 4 were strong lo-fi picks | Yes — correctly noted the narrow energy range |
| 2 | High-energy gym, no genre | Correct — 4 genres, all high energy | Yes — flagged that slots 3–5 were nearly tied |
| 3 | Mellow reggae, discovery mode | Partial — #1 correct; #2–5 were cross-genre fallbacks | Yes — explained confidence drop, suggested genre_first |
| 4 | High energy + melancholic (conflicting) | Incorrect — mood signal ignored | Yes — explicitly named the conflict and explained why |

The self-critique identified the problem accurately in all four cases, including the one
where the recommendations were wrong.

---

## 8. Collaboration with AI During This Project

AI assistance (GitHub Copilot Inline Chat) was used at two specific points.

**Helpful — `diverse_recommend_songs` in `recommender.py`**

The prompt asked for a greedy re-ranking algorithm that subtracts artist and genre repeat
penalties before each slot selection, with compounding penalties and penalty lines appended
to the reasons list. The AI generated a clean O(k × n) greedy loop that matched the spec
exactly, worked on the first try, and required no structural changes. This was the right
tool: the algorithm is well-defined and deterministic, and the AI translated a precise
natural language spec into working code.

**Flawed — `print_summary_table` in `main.py`**

The prompt asked for a `tabulate` table with multi-line cells — song/artist on separate
lines, score with an ASCII bar, and reasons as bullet points. The AI produced the correct
column structure and `fancy_grid` format, but rendered multi-line cells using `\n` in a
single string, which `tabulate` did not split correctly across terminal widths. Cells
overflowed or collapsed depending on the environment. The fix required manually constructing
each cell as a pre-formatted string and testing it across terminal widths — work the
suggestion had implied was unnecessary.

**Lesson:** AI-generated display code should always be tested in the actual output
environment. Visual formatting bugs do not show up in code review or static analysis.

AI was also used for exploratory conversations during design — asking it to critique the
guardrail placement strategy or explain tradeoffs between RAG approaches. These were
treated as a sounding board, not as authoritative; every design decision was verified
against actual test results.

---

## 9. Intended Use and Non-Intended Use

**Intended use:** A learning project demonstrating how to build a reliable AI pipeline —
guardrails, RAG, function calling, agentic self-critique, and structured evaluation —
using a small, inspectable domain (music recommendation).

**Not intended for:** Real users or a live product. The 18-song catalog is too small to be
useful in practice. Features were assigned by hand, not measured from audio. The system has
no memory of past listening, no feedback loop, and no ability to discover new preferences
over time. Do not use it for actual music recommendations outside of a learning context.

---

## 10. Ideas for Improvement

1. **Genre adjacency scoring.** Give partial credit for sonically related genres (e.g.,
   rock→metal earns 1.5 pts instead of 0). This fixes the biggest intuition failure found
   in testing.

2. **Valence as a second numeric signal.** The catalog has valence data (emotional
   positivity) that is never used in scoring. A valence proximity score would help separate
   sad from happy songs when mood labels are ambiguous.

3. **Multi-turn conversation memory.** The CLI resets preferences on every query. Letting
   users say "keep the same mood but change the genre" would make the interaction feel
   natural.

4. **Vector-store RAG.** The static dict knowledge base cannot be updated without a code
   change. Replacing it with a small vector store (e.g., ChromaDB) would allow catalog and
   knowledge updates without touching source code.

5. **Larger catalog with real audio features.** Replacing hand-assigned feature values with
   data from a real audio analysis API (e.g., Spotify Audio Features) would make the energy
   and acousticness signals far more reliable.

---

## 11. Personal Reflection

Building VibeFinder AI reinforced that **AI capability and AI reliability are separate
problems, and reliability is harder.** The Groq LLM parses natural language gracefully, but
without guardrails a hallucinated field value would silently produce meaningless results.
Reliability comes from pipeline design — catching and surfacing failures — not from the
model being perfect.

The RAG layer showed that **grounded generation is more reliable than prompting for
detail.** Without domain context, LLM explanations were generic. With a short genre/mood
paragraph injected at call time, they became specific and differentiated. A modest retrieval
step produced a larger quality gap than expected.

The self-critique loop showed that **LLMs can reason usefully about their own outputs** —
identifying diversity gaps, suspicious confidence patterns, and when a mode switch would
serve the user better. That feedback is not always acted on automatically, but surfacing it
transparently to the user is a meaningful form of reliability that a pure scoring system
cannot provide.

The hardest part was not writing the AI calls — it was deciding *where* to put the
guardrails and what exactly they should check. Every boundary between a deterministic layer
and a non-deterministic one is a potential failure point, and making each failure produce a
precise, diagnosable error message required thinking carefully about what could go wrong at
each stage, not just what should go right.
