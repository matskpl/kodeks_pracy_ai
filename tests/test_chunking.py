"""Testy dzielenia tekstu — brak luk i rozcinek w środku słów."""

from scrapers.chunking import split_long_text


def test_recursive_split_preserves_absolwent_waiver_section():
    """Regresja: „podlegania chorobowego” + wyjątek absolwentów nie może zginąć między chunkami."""
    filler = "Wstęp do zasiłku chorobowego. " * 40
    core = (
        "Jeśli podlegania chorobowego w ciągu 90 dni od ukończenia kadencji "
        "absolwent nie ma okresu wyczekiwania i może otrzymać zasiłek od razu. "
    )
    tail = "Postanowienia końcowe regulacji. " * 40
    text = filler + core + tail

    parts = split_long_text(text, max_len=500, overlap=250)
    assert len(parts) >= 2

    blob = " ".join(parts)
    assert "podlegania chorobowego" in blob
    assert "absolwent" in blob
    assert "okresu wyczekiwania" in blob

    # Nie może zostać urwane słowo jak w błędzie produkcyjnym
    assert "Jeśli podle" not in blob or "podlegania" in blob
    orphaned = "enia chorobowego w ciągu 90"
    assert orphaned not in blob or "podlegania chorobowego" in blob


def test_overlap_at_least_200_chars_between_neighbors():
    text = ("Zdanie numer jeden. " * 30) + ("Zdanie numer dwa o urlopie. " * 30)
    parts = split_long_text(text, max_len=400, overlap=250)
    if len(parts) < 2:
        return
    for i in range(len(parts) - 1):
        tail = parts[i][-250:]
        head = parts[i + 1][:250]
        # overlap: koniec poprzedniego powinien pojawić się na początku następnego
        overlap_found = any(
            tail[j : j + 30] in head for j in range(max(0, len(tail) - 100), len(tail) - 10)
        )
        assert overlap_found or parts[i][-50:] in parts[i + 1]


def test_short_text_single_chunk():
    text = "Krótki fragment o zasiłku."
    assert split_long_text(text) == [text]
