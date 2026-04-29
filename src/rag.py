"""
Retrieval-Augmented Generation knowledge base for the music recommender.

Three data sources are combined by retrieve_context():
  1. GENRE_DOCS  — sonic characteristics per genre
  2. MOOD_DOCS   — emotional and sonic properties per mood
  3. DECADE_DOCS — era-specific production context per decade

retrieve_context() returns a text snippet injected into LLM prompts so
explanations are grounded in music-domain knowledge rather than relying solely
on the model's general training data.
"""

from typing import Optional

GENRE_DOCS: dict[str, str] = {
    "pop": (
        "Pop music prioritises broad appeal, catchy melodies, and polished production. "
        "Contemporary pop blends EDM, R&B, and hip-hop elements. "
        "Key traits: verse-chorus structure, 100-130 BPM, high danceability, low acousticness."
    ),
    "lofi": (
        "Lo-fi hip hop uses deliberately low-fidelity audio textures—vinyl crackle, "
        "tape hiss, chopped jazz samples—to create a relaxed, nostalgic atmosphere. "
        "Typical BPM: 65-90. Very high acousticness and warmth. Low energy by design."
    ),
    "rock": (
        "Rock spans a wide tonal range from melodic indie to heavy metal. "
        "Characterised by electric guitars, strong drum presence, and 4/4 time. "
        "Energy ranges from 0.4 (soft rock) to 0.98 (death metal)."
    ),
    "metal": (
        "Metal is defined by distorted guitars, complex drumming, and aggressive or "
        "anthemic vocal styles. Sub-genres include thrash, death, black, doom, and power metal. "
        "Typical BPM: 120-200. Very low acousticness, very high energy."
    ),
    "hip-hop": (
        "Hip-hop centres on rhythmic vocal delivery over sampled or produced beats. "
        "Sub-genres: trap (heavy 808s, hi-hat rolls, 130-160 BPM), "
        "boom-bap (punchy drums, jazz samples, 80-100 BPM)."
    ),
    "edm": (
        "Electronic Dance Music is an umbrella for club-oriented styles: house, techno, "
        "trance, dubstep, drum and bass. Defined by 4-on-the-floor kick patterns and "
        "synthesisers. Typical BPM: 120-180. Very low acousticness."
    ),
    "jazz": (
        "Jazz features improvisation, syncopated rhythms, and complex harmonic vocabulary. "
        "Nu-jazz fuses jazz sensibility with electronic production. "
        "Typically acoustic or semi-acoustic; low-to-medium energy; romantic or relaxed mood."
    ),
    "ambient": (
        "Ambient music prioritises atmosphere and texture over melody or rhythm. "
        "Very low energy, slow tempos, high acousticness. "
        "Sub-genres: drone, space ambient, dark ambient."
    ),
    "synthwave": (
        "Synthwave draws on 80s analogue synthesisers, arpeggiated basslines, and "
        "neon-noir aesthetics. Typical BPM: 100-130. Mix of nostalgic and moody qualities."
    ),
    "r&b": (
        "R&B blends soulful vocals with modern production. Neo soul and contemporary "
        "R&B lean on complex chord progressions and introspective lyrics. "
        "Medium energy, high valence, often romantic."
    ),
    "soul": (
        "Soul music emphasises emotional depth and expressive vocals rooted in gospel "
        "tradition. Typically warm, intimate production with live instrumentation."
    ),
    "classical": (
        "Western classical music spans baroque, romantic, and contemporary concert music. "
        "Near-100% acoustic. Wide dynamic range; peaceful or intense depending on period."
    ),
    "country": (
        "Country music blends Southern American folk, bluegrass, and pop. "
        "Storytelling lyrics, acoustic guitar or pedal steel, moderate tempo."
    ),
    "reggae": (
        "Reggae originated in Jamaica; defined by off-beat guitar rhythms (skank), "
        "prominent bass, and themes of social justice or spirituality. "
        "Typical BPM: 70-100. Uplifting and peaceful. High acousticness."
    ),
    "indie pop": (
        "Indie pop occupies the space between guitar-based indie rock and accessible pop hooks. "
        "Characterised by jangly guitars, earnest lyrics, and moderate production budgets."
    ),
}

MOOD_DOCS: dict[str, str] = {
    "happy": (
        "Happy tracks project joy, optimism, and celebration. "
        "High valence (>0.7), often high energy, major-key harmonies."
    ),
    "chill": (
        "Chill music creates a relaxed, undemanding atmosphere—suitable for studying, "
        "reading, or winding down. Low-to-medium energy, smooth textures."
    ),
    "intense": (
        "Intense tracks build tension or urgency through fast tempo, distorted timbres, "
        "or aggressive dynamics. Energy typically >0.80."
    ),
    "focused": (
        "Focused music aids concentration: minimal lyrics, steady BPM, moderate energy. "
        "Often ambient, lofi, or classical in nature."
    ),
    "euphoric": (
        "Euphoric tracks deliver peak emotional highs—festival anthems, uplifting EDM drops. "
        "Very high energy and valence."
    ),
    "moody": (
        "Moody music sits in ambiguous emotional territory—introspective, bittersweet, "
        "or cinematically dark. Medium valence, varied energy."
    ),
    "romantic": (
        "Romantic tracks evoke intimacy, longing, or tenderness. "
        "High valence, smooth production, moderate energy."
    ),
    "melancholic": (
        "Melancholic music explores sadness, nostalgia, or loss with artful restraint. "
        "Low valence, slower tempos, often acoustic or sparse arrangements."
    ),
    "angry": (
        "Angry tracks channel aggression, defiance, or catharsis. "
        "Very high energy, low valence, often distorted or abrasive."
    ),
    "uplifting": (
        "Uplifting music inspires hope or forward momentum without necessarily being euphoric. "
        "Moderate-to-high energy, positive valence."
    ),
    "peaceful": (
        "Peaceful tracks offer calm and serenity. Low energy, high acousticness, gentle dynamics."
    ),
    "nostalgic": (
        "Nostalgic music evokes memories of the past through familiar progressions, "
        "vintage timbres, or lyrical callbacks to earlier times."
    ),
    "relaxed": (
        "Relaxed music sits between peaceful and chill—unhurried, comfortable, easy-listening. "
        "Low-to-medium energy, warm timbre."
    ),
}


DECADE_DOCS: dict[int, str] = {
    1980: (
        "1980s production is defined by gated reverb drums (the Phil Collins 'big drum' sound), "
        "analog synthesisers (Roland Juno-106, Oberheim OB-Xa), and bright, punchy mixing. "
        "Genres that crystallised this decade: new wave, synth-pop, glam metal, early hip-hop."
    ),
    1990: (
        "1990s music split between grunge's raw guitar distortion and alternative rock on one side, "
        "and polished R&B/pop on the other. CD-era production prized dynamic clarity; "
        "lo-fi cassette aesthetics emerged as a counter-reaction. "
        "BPM range widened dramatically with the rise of rave culture (house, jungle, drum and bass)."
    ),
    2000: (
        "2000s pop and R&B leaned on Pro Tools digital recording, Auto-Tune pitch correction, "
        "and large-room reverb on lead vocals. Guitar-driven post-punk revival and indie rock "
        "coexisted with the rise of crunk and hyphy in hip-hop. "
        "Production became louder due to the loudness wars."
    ),
    2010: (
        "2010s production favoured heavy side-chain compression, drop structures in EDM, "
        "trap hi-hat rolls (Roland TR-808 resampled), and a warm vintage revival in indie pop. "
        "Streaming loudness normalisation (−14 LUFS target) began reshaping how engineers mixed, "
        "pushing back against the loudness wars of the 2000s."
    ),
    2020: (
        "2020s music is shaped by bedroom production: laptop DAWs, royalty-free sample packs, "
        "TikTok-driven viral hooks, and lo-fi aesthetics returning at mass scale. "
        "Hyperpop, Afrobeats, and genre-blended playlists dominate streaming. "
        "Spatial audio (Dolby Atmos) is the emerging mixing standard for premium releases."
    ),
}


def retrieve_context(genre: str = "", mood: str = "", decade: int = 0) -> str:
    """
    Return knowledge-base snippets for the given genre, mood, and/or decade.

    Draws from three data sources (GENRE_DOCS, MOOD_DOCS, DECADE_DOCS).
    decade=0 means no era preference — decade context is skipped.
    Returns a fallback message only when all three sources produce nothing.
    """
    parts: list[str] = []
    g = genre.lower().strip()
    m = mood.lower().strip()
    if g and g in GENRE_DOCS:
        parts.append(f"[Genre: {g}]\n{GENRE_DOCS[g]}")
    if m and m in MOOD_DOCS:
        parts.append(f"[Mood: {m}]\n{MOOD_DOCS[m]}")
    if decade and decade in DECADE_DOCS:
        parts.append(f"[Decade: {decade}s]\n{DECADE_DOCS[decade]}")
    return "\n\n".join(parts) if parts else "No specific genre/mood context available."


def list_known_genres() -> list[str]:
    return sorted(GENRE_DOCS.keys())


def list_known_moods() -> list[str]:
    return sorted(MOOD_DOCS.keys())


def list_known_decades() -> list[int]:
    return sorted(DECADE_DOCS.keys())
