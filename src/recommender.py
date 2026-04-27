from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import csv

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # --- Advanced features (default values keep existing tests passing) ---
    popularity: int = 50        # 0–100 mainstream appeal score
    release_decade: int = 2010  # e.g. 1990, 2000, 2010, 2020
    mood_tags: str = ""         # comma-separated detailed mood descriptors
    explicit: int = 0           # 1 = contains explicit content, 0 = clean
    subgenre: str = ""          # finer genre classification


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    # --- Advanced preference fields (all optional with sensible defaults) ---
    target_popularity: int = 50         # preferred mainstream level (0–100)
    preferred_decade: int = 0           # 0 = no era preference
    desired_mood_tags: list = field(default_factory=list)  # e.g. ["euphoric","nostalgic"]
    allow_explicit: bool = True         # False = penalise explicit songs
    preferred_subgenre: str = ""        # e.g. "lo-fi hip hop", "death metal"


# ---------------------------------------------------------------------------
# Scoring modes — Strategy pattern
#
# Each mode is a weight dict that controls how much every signal is worth.
# Passing a different mode to score_song / recommend_songs changes the entire
# ranking without touching any other logic.
#
# Key          Meaning
# --------     -----------------------------------------------------------
# genre        Points for an exact genre match
# mood         Points for an exact mood match
# energy       Max points for a perfect energy proximity match
# acoustic     Bonus when acoustic/electronic preference is satisfied
# popularity   Max points for a perfect popularity proximity match
# decade       Points for an exact decade match (half awarded if 1 off)
# mood_tag     Points per matching detailed mood tag (max 3 tags)
# explicit_penalty  Penalty when user disallows explicit and song is explicit
# subgenre     Points for an exact subgenre match
# ---------------------------------------------------------------------------
SCORING_MODES: Dict[str, Dict[str, float]] = {

    # All signals weighted at their original values.  Use this as the baseline.
    "balanced": {
        "genre":            2.0,
        "mood":             1.0,
        "energy":           1.5,
        "acoustic":         0.5,
        "popularity":       0.75,
        "decade":           1.0,
        "mood_tag":         0.4,
        "explicit_penalty": -2.0,
        "subgenre":         1.0,
    },

    # Genre and subgenre dominate — all other signals are reduced.
    # Best for listeners who never stray from a single genre.
    "genre_first": {
        "genre":            4.0,
        "mood":             0.5,
        "energy":           1.0,
        "acoustic":         0.25,
        "popularity":       0.5,
        "decade":           0.5,
        "mood_tag":         0.2,
        "explicit_penalty": -2.0,
        "subgenre":         2.0,
    },

    # Mood label and detailed mood tags dominate — genre matters less.
    # Best for emotion-driven listeners ("I want something intense right now").
    "mood_first": {
        "genre":            1.0,
        "mood":             3.0,
        "energy":           1.0,
        "acoustic":         0.25,
        "popularity":       0.5,
        "decade":           0.5,
        "mood_tag":         0.8,
        "explicit_penalty": -2.0,
        "subgenre":         0.5,
    },

    # Energy and acoustic character dominate — pure sonic match.
    # Best for workout or focus playlists where vibe > genre.
    "energy_focused": {
        "genre":            0.5,
        "mood":             0.5,
        "energy":           4.0,
        "acoustic":         1.5,
        "popularity":       0.5,
        "decade":           0.5,
        "mood_tag":         0.2,
        "explicit_penalty": -2.0,
        "subgenre":         0.25,
    },

    # Popularity and era lead; genre/mood are heavily reduced.
    # Designed to break filter bubbles and surface unexpected but fitting songs.
    "discovery": {
        "genre":            0.5,
        "mood":             0.5,
        "energy":           1.5,
        "acoustic":         0.5,
        "popularity":       2.0,
        "decade":           2.0,
        "mood_tag":         1.0,
        "explicit_penalty": -1.0,
        "subgenre":         0.25,
    },
}


def get_max_score(mode: str = "balanced") -> float:
    """Return the theoretical maximum score for a mode (assumes 3 mood-tag matches)."""
    w = SCORING_MODES[mode]
    return round(
        w["genre"] + w["mood"] + w["energy"] + w["acoustic"] +
        w["popularity"] + w["decade"] + 3 * w["mood_tag"] + w["subgenre"],
        2,
    )


def score_song(
    user_prefs: Dict, song: Dict, weights: Dict = None
) -> Tuple[float, List[str]]:
    """
    Return (score, reasons) for one song against a user preference dict.

    weights — a scoring-mode dict from SCORING_MODES (default: balanced).
    The same scoring logic runs for every mode; only the point values change.
    """
    if weights is None:
        weights = SCORING_MODES["balanced"]

    score   = 0.0
    reasons: List[str] = []

    # --- Genre match ---
    if song["genre"] == user_prefs.get("genre", ""):
        pts = weights["genre"]
        score += pts
        reasons.append(f"genre match (+{pts})")

    # --- Mood match ---
    if song["mood"] == user_prefs.get("mood", ""):
        pts = weights["mood"]
        score += pts
        reasons.append(f"mood match (+{pts})")

    # --- Energy proximity (continuous, 0–max) ---
    # Closer energy = more points; perfect match = full weight.
    target_energy = float(user_prefs.get("energy", 0.5))
    energy_gap    = abs(target_energy - song["energy"])
    energy_pts    = round(weights["energy"] * (1.0 - energy_gap), 2)
    score        += energy_pts
    reasons.append(f"energy proximity (+{energy_pts})")

    # --- Acoustic preference ---
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))
    if likes_acoustic and song["acousticness"] >= 0.6:
        pts = weights["acoustic"]
        score += pts
        reasons.append(f"acoustic sound matches preference (+{pts})")
    elif not likes_acoustic and song["acousticness"] < 0.4:
        pts = weights["acoustic"]
        score += pts
        reasons.append(f"electronic sound matches preference (+{pts})")

    # --- Popularity proximity (continuous, 0–max) ---
    # Mirrors energy proximity: 1 - (gap / 100) scaled by the mode weight.
    target_pop = int(user_prefs.get("popularity", 50))
    song_pop   = int(song.get("popularity", 50))
    pop_gap    = abs(target_pop - song_pop) / 100.0
    pop_pts    = round(weights["popularity"] * (1.0 - pop_gap), 2)
    score     += pop_pts
    reasons.append(f"popularity proximity (+{pop_pts})")

    # --- Decade match (step function) ---
    # Exact match = full decade weight; one decade off = half; further = 0.
    # Skipped when either value is 0 (no preference set).
    preferred_decade = int(user_prefs.get("decade", 0))
    song_decade      = int(song.get("release_decade", 0))
    if preferred_decade > 0 and song_decade > 0:
        decade_diff = abs(preferred_decade - song_decade)
        if decade_diff == 0:
            pts = weights["decade"]
            score += pts
            reasons.append(f"decade match ({song_decade}s) (+{pts})")
        elif decade_diff == 10:
            pts = round(weights["decade"] * 0.5, 2)
            score += pts
            reasons.append(f"near decade match ({song_decade}s, 10yr off) (+{pts})")

    # --- Detailed mood tags (additive per tag) ---
    # Each tag in the user's desired list that appears in the song's list earns points.
    desired_tags = user_prefs.get("mood_tags", [])
    if isinstance(desired_tags, str):
        desired_tags = [t.strip() for t in desired_tags.split(",") if t.strip()]
    song_tags_raw = song.get("mood_tags", "")
    song_tags     = [t.strip() for t in song_tags_raw.split(",") if t.strip()]
    matching_tags = [t for t in desired_tags if t in song_tags]
    tag_pts       = round(len(matching_tags) * weights["mood_tag"], 2)
    if tag_pts > 0:
        score += tag_pts
        reasons.append(f"mood tags {matching_tags} (+{tag_pts})")

    # --- Explicit content filter (penalty) ---
    allow_explicit = bool(user_prefs.get("allow_explicit", True))
    if not allow_explicit and int(song.get("explicit", 0)) == 1:
        pts = weights["explicit_penalty"]
        score += pts
        reasons.append(f"explicit content penalty ({pts})")

    # --- Subgenre match (all-or-nothing bonus) ---
    preferred_subgenre = user_prefs.get("subgenre", "")
    if preferred_subgenre and song.get("subgenre", "") == preferred_subgenre:
        pts = weights["subgenre"]
        score += pts
        reasons.append(f"subgenre match '{preferred_subgenre}' (+{pts})")

    return round(score, 4), reasons


# ---------------------------------------------------------------------------
# OOP interface — Recommender class (used by tests/test_recommender.py)
# ---------------------------------------------------------------------------

def _profile_to_prefs(user: UserProfile) -> Dict:
    """Convert a UserProfile dataclass to the dict format score_song expects."""
    return {
        "genre":          user.favorite_genre,
        "mood":           user.favorite_mood,
        "energy":         user.target_energy,
        "likes_acoustic": user.likes_acoustic,
        "popularity":     user.target_popularity,
        "decade":         user.preferred_decade,
        "mood_tags":      user.desired_mood_tags,
        "allow_explicit": user.allow_explicit,
        "subgenre":       user.preferred_subgenre,
    }

def _song_to_dict(song: Song) -> Dict:
    """Convert a Song dataclass to the dict format score_song expects."""
    return {
        "id":             song.id,
        "title":          song.title,
        "artist":         song.artist,
        "genre":          song.genre,
        "mood":           song.mood,
        "energy":         song.energy,
        "tempo_bpm":      song.tempo_bpm,
        "valence":        song.valence,
        "danceability":   song.danceability,
        "acousticness":   song.acousticness,
        "popularity":     song.popularity,
        "release_decade": song.release_decade,
        "mood_tags":      song.mood_tags,
        "explicit":       song.explicit,
        "subgenre":       song.subgenre,
    }


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        """Store the song catalog this recommender will rank against."""
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5, mode: str = "balanced") -> List[Song]:
        """Return the top-k songs ranked by descending score for the given mode."""
        prefs   = _profile_to_prefs(user)
        weights = SCORING_MODES[mode]
        scored  = [
            (song, score_song(prefs, _song_to_dict(song), weights)[0])
            for song in self.songs
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(
        self, user: UserProfile, song: Song, mode: str = "balanced"
    ) -> str:
        """Return a plain-language explanation for why a song was recommended."""
        prefs   = _profile_to_prefs(user)
        weights = SCORING_MODES[mode]
        total, reasons = score_song(prefs, _song_to_dict(song), weights)
        return f"Score {total:.2f} — " + "; ".join(reasons)


# ---------------------------------------------------------------------------
# Functional interface used by src/main.py
# ---------------------------------------------------------------------------

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dicts."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append({
                "id":             int(row["id"]),
                "title":          row["title"],
                "artist":         row["artist"],
                "genre":          row["genre"],
                "mood":           row["mood"],
                "energy":         float(row["energy"]),
                "tempo_bpm":      float(row["tempo_bpm"]),
                "valence":        float(row["valence"]),
                "danceability":   float(row["danceability"]),
                "acousticness":   float(row["acousticness"]),
                "popularity":     int(row["popularity"]),
                "release_decade": int(row["release_decade"]),
                "mood_tags":      row["mood_tags"],
                "explicit":       int(row["explicit"]),
                "subgenre":       row["subgenre"],
            })
    return songs


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = "balanced"
) -> List[Tuple[Dict, float, List[str]]]:
    """
    Score every song and return the top-k as (song, score, reasons) tuples.

    mode — one of the keys in SCORING_MODES (default: "balanced").
    Changing the mode rewrites the weight table without touching any other logic.
    """
    weights = SCORING_MODES[mode]
    scored  = [
        (song, *score_song(user_prefs, song, weights))
        for song in songs
    ]
    return sorted(scored, key=lambda entry: entry[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Diversity penalty constants
#
# Inline Chat design prompt used to specify this logic:
#   "Add a diverse_recommend_songs function that scores all songs first, then
#    greedily builds the top-k list one slot at a time. Before each pick,
#    subtract ARTIST_REPEAT_PENALTY from any candidate whose artist is already
#    in the selected list, and subtract GENRE_REPEAT_PENALTY per occurrence of
#    their genre already selected. Penalties compound: a third song from the
#    same artist loses twice the single-repeat penalty. Append penalty lines to
#    the reasons list so the output stays fully transparent."
# ---------------------------------------------------------------------------
ARTIST_REPEAT_PENALTY: float = -1.5  # per occurrence of artist in selected list
GENRE_REPEAT_PENALTY:  float = -0.5  # per occurrence of genre  in selected list


def diverse_recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: str = "balanced",
    artist_penalty: float = ARTIST_REPEAT_PENALTY,
    genre_penalty:  float = GENRE_REPEAT_PENALTY,
) -> List[Tuple[Dict, float, List[str]]]:
    """
    Greedy re-ranking that enforces artist and genre diversity.

    Algorithm:
      1. Score every song with the chosen mode (initial scores, no penalties).
      2. Build the result list one slot at a time.
      3. Before each pick, compute an adjusted score for every remaining
         candidate:  adj = raw_score
                          + artist_penalty × (times artist already selected)
                          + genre_penalty  × (times genre  already selected)
      4. The candidate with the highest adjusted score fills the next slot.
         Its penalty lines are appended to its reasons list so the caller
         can see exactly why it was promoted or demoted.

    This is O(k × n) — fast enough for any realistic catalog size.
    """
    weights = SCORING_MODES[mode]

    # Step 1 — score every song once; keep raw scores and reasons
    pool: List[Tuple[Dict, float, List[str]]] = [
        (song, *score_song(user_prefs, song, weights))
        for song in songs
    ]
    # Sort pool so ties are broken by raw score (highest first)
    pool.sort(key=lambda x: x[1], reverse=True)

    selected: List[Tuple[Dict, float, List[str]]] = []

    # Step 2-4 — greedy selection with diversity adjustments
    while len(selected) < k and pool:
        # Count how many times each artist / genre already appears in selected
        artist_counts: Dict[str, int] = {}
        genre_counts:  Dict[str, int] = {}
        for sel_song, _, _ in selected:
            a = sel_song["artist"]
            g = sel_song["genre"]
            artist_counts[a] = artist_counts.get(a, 0) + 1
            genre_counts[g]  = genre_counts.get(g, 0) + 1

        best_adj   = float("-inf")
        best_idx   = 0
        best_extra: List[str] = []

        for i, (song, raw_score, _) in enumerate(pool):
            adj_score = raw_score
            extra: List[str] = []

            a_reps = artist_counts.get(song["artist"], 0)
            if a_reps > 0:
                pen = round(artist_penalty * a_reps, 2)
                adj_score += pen
                extra.append(f"artist repeat penalty ({pen})")

            g_reps = genre_counts.get(song["genre"], 0)
            if g_reps > 0:
                pen = round(genre_penalty * g_reps, 2)
                adj_score += pen
                extra.append(f"genre repeat penalty ({pen})")

            if adj_score > best_adj:
                best_adj  = adj_score
                best_idx  = i
                best_extra = extra

        song, raw_score, raw_reasons = pool.pop(best_idx)
        selected.append((song, round(best_adj, 4), raw_reasons + best_extra))

    return selected
