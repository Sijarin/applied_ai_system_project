# VibeFinder AI — Agentic Music Recommender

> A natural language music recommendation system built on a content-based scoring engine,
> extended with a Groq LLM agent, retrieval-augmented generation, multi-stage guardrails,
> and a self-critiquing agentic loop.

**Extra credit features implemented:** RAG Enhancement · Agentic Workflow Enhancement · Fine-Tuning / Specialization · Test Harness

---

## Original Project (Module 3): Music Recommender Simulator

The original project, **Music Recommender Simulator**, was built during Module 3 as a
content-based music recommender. Given a structured user taste profile (genre, mood, energy
level, acoustic preference), it scores every song in a CSV catalog using a weighted rule
system and returns the top-5 matches with a plain-language explanation of each score. The
system exposed key recommender tradeoffs — genre dominance, the absence of genre adjacency,
and filter bubble dynamics — through six adversarial test profiles run against an 18-song
catalog.

---

## What This Project Does and Why It Matters

VibeFinder AI extends the original recommender into a conversational agent. Instead of
filling out a preference form, a user types plain English — *"something chill and acoustic
for studying, lo-fi vibes"* — and the system handles the rest: parsing intent, validating
it, querying the catalog, explaining results, and then evaluating its own output before
presenting it to the user.

This matters because it demonstrates a complete applied AI pipeline:

- **Natural language understanding** via LLM function calling
- **Grounded generation** via retrieval-augmented context
- **Safety** via layered input and output guardrails
- **Transparency** via per-song scoring reasons and confidence labels
- **Self-correction** via an agentic self-critique loop
- **Auditability** via structured JSON event logging

---

## Architecture Overview

The system is organized as a linear pipeline with two guardrail checkpoints, one RAG
injection, and one agentic feedback loop at the end.

```
User (natural language)
      │
      ▼
┌─────────────────┐
│  Input Guardrail│  guardrails.py · sanitize_text()
└────────┬────────┘
         │ clean text
         ▼
┌─────────────────┐     ┌──────────────────────────────┐
│   AI Parser     │◄────│  RAG Knowledge Base (rag.py)  │
│  ai_agent.py    │     │  GENRE_DOCS · MOOD_DOCS       │
│  Groq LLM +     │     └──────────────────────────────┘
│  function call  │
└────────┬────────┘
         │ structured prefs dict
         ▼
┌─────────────────┐
│  Input Guardrail│  guardrails.py · validate_preferences()
└────────┬────────┘
         │ validated prefs
         ▼
┌─────────────────┐     ┌──────────────────┐
│   Recommender   │◄────│  data/songs.csv  │
│  recommender.py │     └──────────────────┘
│  score + MMR    │
└────────┬────────┘
         │ top-5 (song, score, reasons)
         ▼
┌─────────────────┐
│ Output Guardrail│  guardrails.py · validate_recommendations()
└────────┬────────┘
         │ validated results
         ▼
┌─────────────────┐
│   Evaluator     │  evaluation.py
│                 │  confidence_score() · evaluate_recommendations()
└────────┬────────┘
         │ scored results + aggregate metrics
         ▼
┌─────────────────┐     ┌──────────────────────────────┐
│  AI Explainer   │◄────│  RAG context injected again  │
│  ai_agent.py    │     └──────────────────────────────┘
│  Groq LLM       │
└────────┬────────┘
         │ prose explanations
         ▼
  ┌─────────────┐
  │ Display to  │  ← Human reads results here
  │    User     │
  └──────┬──────┘
         ▼
┌─────────────────┐
│  AI Self-Critique│  ai_agent.py · self_critique()
│  (agentic loop) │  Groq LLM evaluates its own output
└────────┬────────┘
         │ critique shown to user
         ▼
┌─────────────────┐
│     Logger      │  logs/events.jsonl · logs/agent.log
└─────────────────┘


Testing layer (human-in-the-loop verification):
  tests/test_guardrails.py  ──►  Input + Output Guardrails
  tests/test_rag.py         ──►  RAG Knowledge Base
  tests/test_evaluation.py  ──►  Confidence scoring + metrics
```

**Key structural points:**

- RAG context is injected at two points: into the LLM parser prompt and into the explainer
  prompt, so both calls are grounded in the same music knowledge base.
- Guardrails appear before the LLM (raw text check), after the LLM (structured prefs
  validation), and after the recommender (output shape check) — the AI calls are sandwiched.
- The self-critique is what makes this "agentic": the LLM reflects on its own output before
  the user acts on it, surfacing diversity gaps or mismatches autonomously.

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/sijarindhakal/applied_ai_system_project.git
cd applied_ai_system_project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

### 5. Run the original recommender (no API key needed)

```bash
python -m src.main
```

### 6. Run the agentic CLI (requires Groq API key)

```bash
python -m src.agent_cli
```

### 7. Run the test suite

```bash
pytest
```

---

## Sample Interactions

The examples below show real agentic CLI sessions.

---

### Example 1 — Chill study session

**Input:**
```
You: something chill and acoustic for studying, lo-fi vibes, nothing explicit
```

**Output:**
```
==============================================================
  Recommendations — BALANCED mode
==============================================================

  [Metrics]  avg confidence: 87%  |  genre diversity: 2
             artist diversity: 4  |  high-confidence: 4/5

  #1  Library Rain — Paper Lanterns
       Genre: lofi  |  Mood: chill  |  BPM: 72.0
       Score: 8.61/9.45  [##################--]  [high confidence]
       Why: genre match (+2.0); mood match (+1.0); energy proximity (+1.46);
            acoustic sound matches preference (+0.5); ...
       AI:  Library Rain wraps you in warm vinyl crackle and gentle piano
            loops tuned for deep focus. The 72 BPM tempo and near-zero
            energy profile make it ideal background music that never intrudes.

  #2  Midnight Coding — LoRoom
       ...

  [AI Self-Critique]
  The top four results are genuinely strong lo-fi picks with consistent
  energy and acoustic character. The fifth slot (ambient/jazz) diverges
  slightly in genre but earns its place through low energy proximity —
  worth keeping for variety. No artist appears twice, which is good.
  One note: all five results share a similarly low energy band (0.28–0.45);
  a user who wanted slight variation in intensity might feel the list is
  too uniform.
```

---

### Example 2 — High-energy workout, no genre constraint

**Input:**
```
You: I need something intense and high energy for the gym, don't care about genre
```

**Output:**
```
==============================================================
  Recommendations — BALANCED mode
==============================================================

  [Metrics]  avg confidence: 79%  |  genre diversity: 4
             artist diversity: 5  |  high-confidence: 3/5

  #1  Storm Runner — Voltline
       Genre: rock  |  Mood: intense  |  BPM: 148.0
       Score: 7.20/9.45  [###############-----]  [high confidence]
       Why: mood match (+1.0); energy proximity (+1.49); ...
       AI:  Storm Runner delivers relentless distorted guitars and a
            locomotive drum pattern that peaks at the two-minute mark.
            At 148 BPM and 0.96 energy, it is one of the hardest-hitting
            tracks in the catalog for sustained cardio work.

  #2  Iron Curtain — Dreadnought
       Genre: metal  |  Mood: angry  |  BPM: 160.0
       Score: 6.91/9.45  [##############------]  [high confidence]
       ...

  [AI Self-Critique]
  The list covers four genres (rock, metal, hip-hop, EDM) which is strong
  diversity for a genre-agnostic workout query. All five songs sit above
  0.88 energy, which directly matches the intent. The one concern is that
  slots #3–5 have nearly identical scores (6.4–6.5), so the ranking among
  them is somewhat arbitrary — a user who dislikes hip-hop would benefit
  from using the genre_first or energy_focused mode instead.
```

---

### Example 3 — Guardrail triggers on bad input

**Input:**
```
You: I want vibing music
```

**Output:**
```
  [Guardrail] Unknown mood 'vibing'. Known: angry, chill, euphoric,
  focused, happy, intense, melancholic, moody, nostalgic, peaceful,
  relaxed, romantic, uplifting.
```

The LLM parsed "vibing" as mood=vibing, which the guardrail correctly rejected before
any recommendation was attempted, prompting the user to rephrase.

---

### Example 4 — Discovery mode breaks the filter bubble

**Input:**
```
You: mode discovery
You: mellow reggae, acoustic, something uplifting
```

**Output:**
```
  Mode set to: discovery

==============================================================
  Recommendations — DISCOVERY mode
==============================================================

  [Metrics]  avg confidence: 61%  |  genre diversity: 4
             artist diversity: 5  |  high-confidence: 1/5

  #1  Island Morning — Reef Roots
       Genre: reggae  |  Mood: uplifting  |  BPM: 88.0
       Score: 5.90/8.75  [#############-------]  [high confidence]
       ...

  #2  Dusty Highway — The Ramblers
       Genre: country  |  Mood: uplifting  |  BPM: 94.0
       Score: 3.80/8.75  [########------------]  [medium confidence]
       AI:  Dusty Highway shares reggae's mid-tempo acoustic warmth
            and uplifting lyrical tone even though it crosses into
            country territory. The shared folk-acoustic DNA makes it
            a reasonable cross-genre discovery pick.

  [AI Self-Critique]
  Discovery mode is working as intended: only one song matches the
  genre exactly, and the rest are pulled from country, soul, and indie
  pop based on era, popularity proximity, and acoustic character.
  The confidence scores drop sharply after #1 — this is expected in
  discovery mode and reflects genuine uncertainty, not a system error.
  A user who only wanted reggae should switch back to genre_first mode.
```

---

## Design Decisions

### 1. Groq instead of OpenAI
Groq's inference speed (typically < 1 second) keeps the interactive loop responsive.
The llama-3.3-70b-versatile model is capable enough for structured extraction and
prose generation without requiring a paid tier during development.

### 2. Two RAG injections, same knowledge base
RAG context is injected into both the parser prompt and the explainer prompt. This costs
nothing extra (the knowledge base is a static dict, not a vector store) and ensures the
explanations reference real music-domain facts rather than generic LLM outputs.

### 3. Guardrails at three stages, not one
A single output guardrail at the end would be easier but weaker. Rejecting bad input
before the LLM call saves an API round-trip. Rejecting bad LLM-generated prefs before
the recommender prevents silent scoring errors. Checking the recommender output catches
edge cases (empty catalog slice). The cost is three small validation functions; the
benefit is that every failure has a precise, diagnosable error message.

### 4. Self-critique as the agentic loop
Rather than running a multi-turn "plan → execute → reflect → re-plan" loop (which would
be expensive and harder to test), the agentic behavior is focused: the LLM evaluates its
own final output and surfaces concerns the user can act on. This keeps latency low and
the control flow deterministic while still demonstrating the core agentic pattern.

### 5. MMR-style diversity penalty in the recommender
Without a diversity penalty, the same artist appeared twice in the top-5 for multiple
profiles. The greedy re-ranking algorithm (score all songs once, then pick greedily with
artist/genre repeat penalties) adds diversity at O(k × n) cost and makes the penalty
visible in the reasons list, keeping the system fully transparent.

### Trade-offs

| Decision | What was gained | What was given up |
|---|---|---|
| Static RAG dict | Zero latency, no vector DB dependency | Cannot update knowledge without code changes |
| Groq function calling for parsing | Structured output with no regex | Requires valid API key; fails on network error |
| Three guardrail stages | Precise failure messages at each boundary | More code paths to maintain |
| Self-critique (not multi-turn) | Low cost, deterministic flow | Cannot autonomously fix a bad recommendation |
| CSV catalog | Simple, inspectable, version-controlled | No real-time data; limited to 18 songs |

---

## Testing Summary

**42 out of 42 automated tests pass.** Confidence scores averaged 79–87% on well-matched
queries; dropped to ~61% in discovery mode (expected — the mode is designed to surface
uncertain cross-genre picks). Error handling logged and recovered cleanly from every tested
failure case. Human review of 4 live sessions found recommendations matched intent in 3 of 4
cases; the fourth (a conflicting high-energy + melancholic query) surfaced a known scoring
limitation that the self-critique correctly identified.

---

### 1. Automated Tests

```
pytest tests/ -v
42 passed in 0.04s
```

| File | Tests | What is covered |
|---|---|---|
| `tests/test_guardrails.py` | 20 | `sanitize_text` (empty, too long, non-string, whitespace stripping), `validate_preferences` (all 9 fields, boundary values, normalization, defaults), `validate_recommendations` (shape, empty list, bad score type) |
| `tests/test_evaluation.py` | 12 | `confidence_score` at all three thresholds and edge cases (zero max, ratio cap), `evaluate_recommendations` for diversity counts, explicit count, and average confidence |
| `tests/test_rag.py` | 8 | `retrieve_context` for known genre, known mood, both combined, unknown inputs, empty inputs, case-insensitivity, and the sorted genre/mood list helpers |
| `tests/test_recommender.py` | 2 | `Recommender.recommend` returns songs sorted by descending score; `explain_recommendation` returns a non-empty string |

The test suite covers every deterministic layer in the pipeline. LLM calls are intentionally
excluded — they are non-deterministic and require a live API key, so they are validated at
runtime by the guardrails instead.

**Two real bugs were caught by tests before they reached the LLM:**

- `validate_preferences` was silently accepting `decade=2025` (not a round decade). The test
  `test_invalid_decade_raises` exposed this; the fix added the `% 10 != 0` check.
- `mood_tags` passed as a comma-separated string (`"euphoric,danceable"`) was not being split
  into a list. `test_mood_tags_as_string_converted` caught this; the fix added the string
  split branch in `validate_preferences`.

---

### 2. Confidence Scoring

Every recommendation is assigned a confidence label based on how close its score is to the
theoretical maximum for the active scoring mode:

| Label | Threshold | Meaning |
|---|---|---|
| **high** | ≥ 70% of max score | Strong match across multiple signals |
| **medium** | 45–69% of max score | Partial match; some signals missing |
| **low** | < 45% of max score | Weak match; used only to fill remaining slots |

Results from manual testing across four query types:

| Query type | Avg confidence | High-confidence results | Notes |
|---|---|---|---|
| Clear genre + mood (e.g. chill lo-fi) | 87% | 4/5 | Strong catalog coverage for this profile |
| Genre-agnostic, high energy (gym) | 79% | 3/5 | Slots 3–5 are energy-only matches |
| Discovery mode (cross-genre) | 61% | 1/5 | Expected — mode trades confidence for variety |
| Conflicting prefs (high energy + melancholic) | 52% | 1/5 | System optimizes for energy; mood signal not satisfied |

Confidence scores are printed per song and as an aggregate metric after every query, so the
user can see exactly how certain the system is before acting on a recommendation.

---

### 3. Logging and Error Handling

Every agent turn appends structured JSON to `logs/events.jsonl`. Each event captures the
timestamp, event type, and all relevant data so failures can be diagnosed without re-running
the session.

Event types recorded:

| Event | When it fires | What is logged |
|---|---|---|
| `query` | User submits input | raw text, scoring mode |
| `parse_error` | Groq API call fails | exception message |
| `guardrail_block` | Input or prefs fail validation | field name, rejection reason |
| `recommendations` | Recommender returns results | validated prefs, mode, top-5 titles and scores |
| `critique` | Self-critique completes | full critique text |

Example log entry for a guardrail block:

```json
{
  "ts": "2026-04-26T14:22:03.441Z",
  "event": "guardrail_block",
  "error": "Unknown mood 'vibing'. Known: angry, chill, euphoric, ..."
}
```

Human-readable runtime logs are also written to `logs/agent.log` at INFO level. Errors at
the parse, guardrail, and explanation stages are all caught individually so one failure never
crashes the session — the user sees a clean inline message and the loop continues.

---

### 4. Human Evaluation

Four live sessions were reviewed after development:

| Session | Query intent | Result quality | Self-critique accurate? |
|---|---|---|---|
| 1 | Chill lo-fi for studying | Correct — top 4 were strong lo-fi picks | Yes — correctly noted the list was narrow in energy range |
| 2 | High-energy gym, no genre | Correct — 4 genres represented, all high energy | Yes — flagged that slots 3–5 were nearly tied and arbitrary |
| 3 | Mellow reggae, discovery mode | Partial — #1 was correct; #2–5 were cross-genre fallbacks | Yes — explained the confidence drop and suggested genre_first |
| 4 | High energy + melancholic (conflicting) | Incorrect — melancholic mood signal ignored | Yes — explicitly named the conflict and explained why it happened |

The self-critique identified the problem accurately in all four cases, including the one
where the recommendations were wrong. This is the system's strongest reliability signal for
the AI layer: even when the output is imperfect, the critique layer surfaces the right
diagnosis.

---

### What was learned

Testing deterministic code (guardrails, scoring, RAG lookup) with unit tests and testing
non-deterministic code (LLM calls) with runtime guardrails + human review are two
complementary strategies that together give reasonable confidence across the whole pipeline.
Neither alone would be sufficient.

---

## Extra Credit Features

---

### 1. Test Harness (+2)

**File:** [tests/test_harness.py](tests/test_harness.py)

A standalone evaluation script that runs 14 predefined test cases through the deterministic
pipeline layers — guardrails, recommender, evaluator, and RAG retriever — without requiring
a Groq API key.

```
python tests/test_harness.py
```

```
══════════════════════════════════════════════════════════════════════════════
  VibeFinder AI — Evaluation Harness  (no Groq API key required)
══════════════════════════════════════════════════════════════════════════════
  Test Case                       Layer          Result    Detail
  ──────────────────────────────  ─────────────  ────────  ──────────────────────────────────
  sanitize:strips_whitespace      guardrails     ✓ PASS    'chill lo-fi beats'
  sanitize:rejects_empty          guardrails     ✓ PASS    GuardrailError raised correctly
  sanitize:rejects_too_long       guardrails     ✓ PASS    GuardrailError raised for 501-char input
  prefs:valid_full_dict           guardrails     ✓ PASS    genre=pop, mood=happy
  prefs:bad_energy_range          guardrails     ✓ PASS    GuardrailError raised for energy=1.5
  prefs:unknown_genre             guardrails     ✓ PASS    GuardrailError raised for genre='death_polka'
  recommend:top1_genre_match      recommender    ✓ PASS    top result genre='lofi', score=5.65
  recommend:high_confidence       recommender    ✓ PASS    label=high (75% of max_score=8.95)
  recommend:diverse_artists       recommender    ✓ PASS    5/5 unique artists
  recommend:explicit_penalized    recommender    ✓ PASS    'Iron Curtain' rank: #4 → #17 when explicit blocked
  eval:aggregate_metrics          evaluator      ✓ PASS    avg_conf=37%, genre_div=5, artist_div=5
  rag:genre_and_mood              rag            ✓ PASS    397 chars returned, both sections present
  rag:decade_third_source         rag            ✓ PASS    genre + era context combined (445 chars)
  rag:all_three_sources           rag            ✓ PASS    all three data sources present (687 chars)

  RESULT: 14/14 passed — all checks passed ✓
══════════════════════════════════════════════════════════════════════════════
```

Notable results: `recommend:explicit_penalized` shows "Iron Curtain" dropping from rank #4 to
rank #17 when `allow_explicit=False` — confirming the penalty fires correctly. `recommend:diverse_artists` confirms the MMR diversity algorithm produces 5 unique artists from the 18-song catalog.

---

### 2. RAG Enhancement (+2)

**File:** [src/rag.py](src/rag.py)

The original RAG system used two data sources (genre docs and mood docs). A third source,
`DECADE_DOCS`, was added to provide era-specific production context for 1980s–2020s.

```python
retrieve_context(genre="synthwave", mood="moody", decade=1980)
```

**Before (2 sources):**
```
[Genre: synthwave]
Synthwave draws on 80s analogue synthesisers, arpeggiated basslines, and neon-noir aesthetics.
Typical BPM: 100-130. Mix of nostalgic and moody qualities.

[Mood: moody]
Moody music sits in ambiguous emotional territory — introspective, bittersweet, or cinematically dark.
```

**After (3 sources):**
```
[Genre: synthwave]
Synthwave draws on 80s analogue synthesisers, arpeggiated basslines, and neon-noir aesthetics.
Typical BPM: 100-130. Mix of nostalgic and moody qualities.

[Mood: moody]
Moody music sits in ambiguous emotional territory — introspective, bittersweet, or cinematically dark.

[Decade: 1980s]
1980s production is defined by gated reverb drums (the Phil Collins 'big drum' sound),
analog synthesisers (Roland Juno-106, Oberheim OB-Xa), and bright, punchy mixing.
Genres that crystallised this decade: new wave, synth-pop, glam metal, early hip-hop.
```

The added decade context means LLM explanations for 1980s queries now reference specific
instruments (Roland Juno-106, Oberheim OB-Xa) and production techniques (gated reverb)
rather than falling back to generic descriptions. The `retrieve_context()` signature is
backward-compatible — `decade=0` is the default and skips era context entirely.

New tests added to [tests/test_rag.py](tests/test_rag.py):
`test_retrieve_known_decade`, `test_retrieve_all_three_sources`,
`test_retrieve_unknown_decade_ignored`, `test_list_known_decades_sorted` — all pass.

---

### 3. Agentic Workflow Enhancement (+2)

**Files:** [src/ai_agent.py](src/ai_agent.py) · [src/agent_cli.py](src/agent_cli.py)

A `plan_query()` function was added that uses Groq function calling to produce a structured
reasoning plan as a visible Step 0 before parsing begins. This makes the agent's intermediate
reasoning observable.

**Observable output in the CLI:**
```
  [Step 0 — Reasoning Plan]
  Signals    : lofi, studying, acoustic preference, no explicit content, chill vibe
  Mode hint  : mood_first
  Expectation: 3-4 strong lofi/ambient matches likely in 18-song catalog
  Conflicts  : (none)

  [Step 1 — Parsing query]
  ...
  [Step 2 — Validating preferences]
  ...
  [Step 3 — Recommending]
  ...
  [Step 4 — Generating explanations]
  ...
  [Step 5 — Self-critique]
  ...
```

The plan step uses the same Groq function-calling pattern as the parser, returning a typed
dict with four fields: `detected_signals`, `recommended_mode`, `catalog_expectation`, and
`potential_conflicts`. The mode hint is advisory — it does not override the user's explicitly
set mode. If the plan API call fails, the agent degrades gracefully and continues to Step 1.

**Why this qualifies as multi-step reasoning:** the agent now runs four distinct LLM calls
per turn, each with a different role — plan → parse → explain → critique — and each
intermediate output is logged to `logs/events.jsonl` and printed to the terminal.

---

### 4. Fine-Tuning / Specialization (+2)

**File:** [src/ai_agent.py](src/ai_agent.py) — `_EXPLAIN_SYSTEM` constant

Two few-shot examples and a style constraint were added to the explanation prompt. The
examples demonstrate the required sentence structure (sonic-first, emotional-second) and
the style constraint explicitly prohibits generic filler phrases.

**Baseline prompt** (before):
```
"You are a music expert who writes concise, friendly explanations for song recommendations.
Each explanation must be exactly 2 sentences. Ground your explanations in the music-domain
knowledge provided. Be specific about sonic or emotional qualities rather than repeating the score."
```

**Sample baseline output:**
```
Sunrise City is a great pop track that matches your energy level and mood preferences.
It's an upbeat, happy song that would be perfect for when you want something energetic.
```

**Few-shot prompt** (after) — adds 2 grounded examples + style rules:
```
"Sentence 1 must describe a concrete sonic quality — instrumentation, BPM, texture, or
production style. Sentence 2 must describe the emotional payoff for the listener.
Never repeat the genre label or score. Avoid filler phrases like 'perfect for', 'great choice',
or 'you will love this'."

EXAMPLE A — pop/happy, energy=0.82, BPM=118:
  Sunrise City layers shimmering synth arpeggios over a four-on-the-floor kick at 118 BPM,
  giving it an effortlessly danceable momentum. The high-valence chorus delivers a genuine
  rush of optimism that suits a morning commute or a gym warm-up equally well.

EXAMPLE B — lofi/chill, energy=0.35, BPM=72:
  Library Rain pairs a muffled, tape-saturated piano loop with brushed snare hits at 72 BPM,
  creating a deliberately imperfect warmth that modern hi-fi recordings rarely achieve.
  The nostalgic grain of the recording makes it feel like revisiting a favourite coffee shop
  from years ago.
```

**Sample few-shot output:**
```
Sunrise City layers shimmering synth arpeggios over a four-on-the-floor kick at 118 BPM,
giving it an effortlessly danceable momentum. The high-valence chorus delivers a genuine
rush of optimism that pairs naturally with a morning commute or a gym warm-up.
```

**Measurable difference:** the few-shot output names a specific sonic element (synth
arpeggios, BPM value, four-on-the-floor kick), anchors the emotion to a concrete scenario,
and uses zero generic filler. The baseline output uses "great", "perfect for", and repeats
"energy level" and "mood preferences" — all explicitly prohibited by the new style
constraint. The system prompt grew from ~80 tokens to ~400 tokens; Groq latency increased
by roughly 0.2–0.4 seconds per call, which is the only cost.

---

## Responsible AI

### Limitations and Biases

**Genre dominance.** Genre is the highest-weighted signal (2.0 points out of a 9.45 maximum
in balanced mode). A song that perfectly matches a user's mood, energy, and acoustic
preference but belongs to the wrong genre will consistently rank below a mediocre genre
match. This encodes a bias toward genre identity over sonic similarity — it may disadvantage
listeners whose taste genuinely crosses genre lines.

**No genre adjacency.** The scoring rule treats every genre mismatch as equally wrong. Metal
and pop both score zero for a rock listener, even though metal is sonically far closer to
rock than pop is. This is a structural gap, not a weight-tuning problem — no amount of
parameter adjustment fixes a missing feature.

**Catalog underrepresentation.** The 18-song catalog includes only one reggae song, one
jazz song, one country song, and one R&B song. Users whose preferences fall into underserved
genres get one strong match and four fallback results chosen by energy proximity alone. The
system's recommendations reflect who the catalog was built for, not who might actually use it.

**Binary acoustic preference.** The `likes_acoustic` field is a boolean. Real listening
preferences exist on a spectrum — a user might enjoy both acoustic folk and electronic
ambient music depending on context. The binary flag forces a choice and applies it uniformly
across all five recommendation slots.

**Neutral users are poorly served.** Without genre or mood preferences, the highest
achievable confidence score is about 32% of the theoretical maximum. The system was designed
around categorical signals and degrades significantly when those signals are absent.

---

### Could This System Be Misused?

A music recommender carries low direct harm potential, but several misuse vectors are still
worth naming:

**Filter bubble reinforcement.** The genre-first and mood-first scoring modes could be used
to build a playlist experience that never exposes a listener to anything outside their
existing preferences. Over time this narrows taste rather than broadening it. The discovery
mode and diversity penalty were built specifically to counter this, but they require the user
to opt in.

**Catalog bias as product bias.** If this system were deployed at scale with a catalog
controlled by a label or platform, the underrepresentation of certain genres or artists in
the training data would produce systematically worse recommendations for listeners in those
categories — not because the algorithm is unfair by design, but because the data it operates
on is. A real deployment would need catalog audits and genre coverage targets.

**Prompt injection via natural language input.** Because user text is passed directly into
an LLM prompt, a carefully crafted input could attempt to manipulate the model's behavior —
for example, trying to force the parser to return specific song IDs or bypass the explicit
content filter. The current guardrails mitigate this by validating the structured output
rather than trusting the LLM's raw response: even if the LLM were manipulated into
generating a malformed preferences dict, `validate_preferences` would catch and reject it
before it reached the recommender.

**Prevention measures already in place:** input length cap (500 characters), strict
allowlists for genre and mood values, output shape validation, structured event logging for
auditability, and an explicit content flag that the guardrail enforces independently of the
LLM's output.

---

### What Surprised Me During Testing

The most surprising finding was how well the self-critique performed on the cases where the
recommendations were wrong. When a conflicting query (high energy + melancholic mood)
produced results that ignored the mood signal entirely, the self-critique correctly named the
conflict, explained why the scoring logic could not satisfy both signals simultaneously, and
suggested that the user rephrase. The LLM reasoned accurately about the system's own
failure — a more sophisticated form of reliability than simply producing correct output.

The second surprise was how much the RAG injection changed explanation quality. Without the
genre/mood knowledge base, the LLM's explanations were generic and interchangeable. With
even two short paragraphs of domain context, explanations became specific to each song's
sonic character. This was a larger quality gap than expected from such a lightweight
retrieval step.

---

### Collaboration with AI During This Project

AI assistance (GitHub Copilot Inline Chat) was used at two specific points in development,
and both are documented in the source code comments.

**Helpful suggestion — `diverse_recommend_songs` in [recommender.py](src/recommender.py)**

The Inline Chat prompt asked for a greedy re-ranking algorithm that subtracts artist and
genre repeat penalties before each slot selection, with penalties compounding on repeat
appearances and penalty lines appended to the reasons list for transparency. The AI
generated a clean O(k × n) greedy loop that matched the specification exactly. The
suggestion worked on the first try, required no structural changes, and solved the
artist-repeat problem that was visible across multiple test profiles. This was the right
tool for the job: the algorithm is well-defined, the logic is deterministic, and the AI
correctly translated a precise natural language spec into working code.

**Flawed suggestion — `print_summary_table` in [main.py](src/main.py)**

The Inline Chat prompt asked for a `tabulate` table with multi-line cells — song and artist
on separate lines, score with an ASCII bar below it, and all scoring reasons as bullet
points. The AI generated the correct column structure and the correct `fancy_grid` format,
but it initially rendered the multi-line cells using `\n` embedded in a single string, which
tabulate did not split correctly in all terminal widths. The cells overflowed or collapsed
depending on the environment. The fix required manually constructing each cell as a
pre-formatted string and testing it across terminal widths — work the AI's suggestion had
implied was unnecessary. The lesson: AI-generated display code should always be tested in
the actual output environment before trusting it, because visual formatting bugs do not
show up in code review.

---

## Reflection

Building VibeFinder AI reinforced a central lesson: **AI capability and AI reliability are
separate problems, and the second is harder.** The Groq LLM is excellent at parsing natural
language into structured preferences — it handles slang, partial information, and ambiguous
phrasing gracefully. But without guardrails, a hallucinated field value (e.g., mood="vibing")
would silently reach the recommender and produce meaningless results. The system's
reliability comes not from the LLM being perfect but from the pipeline being designed to
catch and surface its failures.

The RAG layer taught a related lesson about **grounded generation**. Without domain context
injected at call time, the LLM's explanations were generic ("this is a chill song with low
energy"). With even a short paragraph about lo-fi hip-hop's vinyl crackle, slow BPM, and
nostalgic character injected into the prompt, the explanations became specific and
differentiated. Grounding the model in retrieved facts is a more reliable path to quality
output than prompting it to "be more detailed."

The self-critique loop showed that **LLMs can reason usefully about their own outputs** —
identifying when a list lacks diversity, when confidence scores are suspicious, or when a
mode switch would serve the user better. That feedback is not always acted on (the system
doesn't automatically fix a bad list), but surfacing it to the user is a meaningful form
of transparency that a pure scoring system cannot provide.

Finally, the original Module 3 work on scoring modes and diversity penalties proved directly
reusable. The agentic layer sits entirely above the recommender — it never touches the
scoring logic. That clean separation between "how to pick songs" and "how to talk about
songs" made the whole system easier to extend, test, and explain.
