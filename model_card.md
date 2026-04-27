# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Goal / Task

VibeFinder suggests songs a listener might enjoy. It takes four inputs — preferred genre, preferred mood, target energy level, and whether the listener likes acoustic sound — and returns the top 5 best-matching songs from a catalog. The goal is to surface songs that feel like a good vibe match, not just a keyword match.

---

## 3. Data Used

The catalog has **18 songs** stored in `data/songs.csv`. Each song has 10 features: id, title, artist, genre, mood, energy (0–1), tempo in BPM, valence (0–1), danceability (0–1), and acousticness (0–1). Genres covered include pop, lofi, rock, ambient, jazz, synthwave, indie pop, r&b, hip-hop, classical, edm, country, metal, soul, and reggae.

**Limits:**
- Most genres have only one or two songs. The catalog is very small.
- All feature values (energy, acousticness, etc.) were assigned by hand — not measured from real audio.
- The catalog only includes Western popular music from roughly 2000–2025. No traditional music or non-English genres are included.

---

## 4. Algorithm Summary

VibeFinder uses a **weighted rule-based scoring system**. Every song in the catalog gets a score. The top 5 scores become the recommendation list.

Here is how each song earns points:

- **Genre match (+2.0 pts):** If the song's genre matches what the user wants, it gets 2 points. This is the biggest reward because genre is the strongest signal of taste.
- **Mood match (+1.0 pt):** If the song's mood label matches, it gets 1 point. Mood matters less than genre because the same mood can appear across very different styles.
- **Energy proximity (up to +1.5 pts):** The closer a song's energy is to the user's target, the more points it earns. A perfect match gives 1.5 points; a big gap gives almost zero.
- **Acoustic preference (+0.5 pts):** If the user likes acoustic sound and the song is very acoustic (or vice versa), the song earns a small bonus.

Maximum possible score is **5.0 points**.

---

## 5. Observed Behavior / Biases

**The mood bonus can override genre reality.** For a "deep intense rock" listener, a pop song tagged "intense" scored higher than a metal song. The system gave the pop song a +1.0 mood bonus that the metal song could not earn. But a rock fan asking for intense music almost certainly wants metal over pop. The system cannot tell those two apart because it only checks exact genre matches — there is no sense of which genres are sonically related.

**Orphaned genres produce random fallbacks.** Ten of fifteen genres have exactly one song. When the one matching song takes the #1 slot, the remaining four spots are filled by energy proximity alone. A reggae fan might end up with lofi and jazz in their top 5 because those happen to have similar energy values.

**Neutral users get poor recommendations.** When no genre or mood is set, the max achievable score drops to about 1.6 out of 5.0. The system is so dependent on the genre and mood bonuses that it can barely separate songs for users who have not stated clear preferences.

---

## 6. Evaluation Process

Six listener profiles were tested: three standard and three adversarial.

**Standard profiles:**
1. High-Energy Pop (genre=pop, mood=happy, energy=0.82, acoustic=no)
2. Chill Lofi (genre=lofi, mood=chill, energy=0.38, acoustic=yes)
3. Deep Intense Rock (genre=rock, mood=intense, energy=0.91, acoustic=no)

**Adversarial profiles:**
4. Conflicting preferences (high energy + melancholic mood — these almost never appear together)
5. Genre orphan (reggae — only one song in the catalog)
6. Dead-centre neutral (no genre or mood set, energy exactly 0.5)

Results were compared to intuition: did the top song feel like the obvious right answer? Were any results surprising or wrong? The weight values were also adjusted once (genre halved, energy doubled) to test whether tuning numbers could fix the Gym Hero vs Iron Curtain ranking problem. It could not.

---

## 7. Intended Use and Non-Intended Use

**Intended use:** This system is a classroom learning tool. It is designed to show how a simple rule-based recommender works and to make the scoring logic fully visible. Every result shows exactly which rules fired and how many points each rule gave.

**Not intended for:** Real users or a live product. The catalog is too small (18 songs) to be useful in practice. The feature values were assigned by hand, not measured from audio. The system has no memory of past listening, no user feedback loop, and no ability to discover new preferences over time. Do not use it to make actual music recommendations outside of a learning context.

---

## 8. Ideas for Improvement

1. **Add genre adjacency scoring.** Instead of a 0-or-2 genre bonus, give partial credit for similar genres. Rock→metal could earn 1.5 points instead of 0. This would fix the biggest intuition failure found in testing.

2. **Use valence as a second numeric feature.** The catalog already has valence data (emotional positivity), but it is never used in scoring. Adding a valence proximity score would help separate sad from happy songs in cases where mood labels alone are too broad.

3. **Add a repeat-artist filter.** The same artist can appear more than once in the top 5. Capping each artist at one result per list would make recommendations feel more varied and useful.

---

## 9. Personal Reflection

My biggest learning moment was realizing that weight-tuning cannot fix a missing feature. I tried reducing the mood weight to fix the Gym Hero vs. Iron Curtain ranking — the result did not change. The system needed genre adjacency, a whole new rule, not just different numbers.

I used AI tools to help structure the scoring logic and trace edge cases. They were useful for explaining why a profile produced a surprising result. But I still had to double-check the suggestions. When the AI recommended lowering MOOD_POINTS, I ran the math myself and found the fix did not work. The AI gave me a direction to test — I had to verify it.

What surprised me most: a four-rule system can still feel like a real recommendation. The Chill Lofi results looked like something a person would actually suggest. The algorithm has no idea what music sounds like, but when the rules line up with how humans think about taste, the output feels natural. It only breaks down when the rules conflict with each other.

If I extended this project, I would add genre adjacency scoring first. Then I would add a simple feedback loop — let the user mark songs as liked or skipped, and update the weights based on their choices. That would make the system learn instead of just matching.
