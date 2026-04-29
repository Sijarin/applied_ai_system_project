from src.rag import retrieve_context, list_known_genres, list_known_moods, list_known_decades


def test_retrieve_known_genre():
    ctx = retrieve_context(genre="pop")
    assert "pop" in ctx.lower()
    assert "[Genre: pop]" in ctx


def test_retrieve_known_mood():
    ctx = retrieve_context(mood="chill")
    assert "[Mood: chill]" in ctx


def test_retrieve_both():
    ctx = retrieve_context(genre="lofi", mood="nostalgic")
    assert "[Genre: lofi]" in ctx
    assert "[Mood: nostalgic]" in ctx


def test_retrieve_unknown_returns_fallback():
    ctx = retrieve_context(genre="polka", mood="")
    assert "No specific genre/mood context available." in ctx


def test_retrieve_empty_inputs_returns_fallback():
    ctx = retrieve_context()
    assert "No specific genre/mood context available." in ctx


def test_list_known_genres_sorted():
    genres = list_known_genres()
    assert genres == sorted(genres)
    assert "pop" in genres
    assert "lofi" in genres


def test_list_known_moods_sorted():
    moods = list_known_moods()
    assert moods == sorted(moods)
    assert "chill" in moods
    assert "happy" in moods


def test_retrieve_case_insensitive_mood():
    ctx_lower = retrieve_context(mood="happy")
    ctx_upper = retrieve_context(mood="HAPPY")
    assert "[Mood: happy]" in ctx_lower


# ── DECADE_DOCS tests (RAG Enhancement — third data source) ───────────────────

def test_retrieve_known_decade():
    ctx = retrieve_context(decade=2010)
    assert "[Decade: 2010s]" in ctx
    assert "2010" in ctx


def test_retrieve_all_three_sources():
    ctx = retrieve_context(genre="pop", mood="happy", decade=2020)
    assert "[Genre: pop]" in ctx
    assert "[Mood: happy]" in ctx
    assert "[Decade: 2020s]" in ctx


def test_retrieve_unknown_decade_ignored():
    # 1970 is not in DECADE_DOCS — should not crash and should still return genre context
    ctx = retrieve_context(genre="jazz", decade=1970)
    assert "[Genre: jazz]" in ctx
    assert "1970" not in ctx


def test_list_known_decades_sorted():
    decades = list_known_decades()
    assert decades == sorted(decades)
    assert 2010 in decades
    assert 2020 in decades
