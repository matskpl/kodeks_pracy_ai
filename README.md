# KodeksPracy AI

Asystent HR i prawny dla **MetalTech Sp. z o.o.** — aplikacja webowa oparta o Kodeks pracy, materiały PIP/ZUS oraz dane kadrowe pracowników. Łączy czat z RAG, kalkulator kadrowy, generator pism PDF i panel pracownika / administratora kadr.

## Funkcje

| Moduł | Opis |
|--------|------|
| **Asystent HR (czat)** | MainSupervisor kieruje pytanie do **LegalRAG** (hybryda Qdrant + BM25 + RRF, reranking). Opcjonalnie **Source Judge** (`agents/answer_judge.py`) ocenia odpowiedź względem fragmentów i żąda poprawki — nie jest to osobny „główny” agent, tylko pętla w RAG |
| **Kalkulator kadrowy** | Urlop, wypowiedzenie, nadgodziny, urlop na żądanie — na podstawie profilu lub parametrów z pytania |
| **Generator pism** | Wnioski i dokumenty HR z eksportem do PDF |
| **Profil pracownika** | Dane urlopowe, staż, nadgodziny — z PostgreSQL |
| **Panel admina** | Przegląd pracowników, obliczenia i pisma w imieniu wybranej osoby |

## Architektura (skrót)

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│  Przeglądarka│────▶│  FastAPI (main.py)                       │
│  HTML + SSE  │     │  Auth · Chat · Kalkulator · PDF · API    │
└─────────────┘     └──────┬─────────────────────┬─────────────┘
                           │                     │
              ┌────────────▼────────────┐   ┌────▼─────────────┐
              │  PostgreSQL           │   │  Qdrant + BM25     │
              │  employees, users     │   │  ISAP / PIP / ZUS  │
              └───────────────────────┘   └────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Google Gemini (pydantic-ai) │
              │  Cohere (embed + rerank)     │
              └─────────────────────────────┘
```

**Agenci (pydantic-ai):**

1. **MainSupervisor** — tylko klasyfikacja intencji (legal_rag / calculator / document / general), bez własnej odpowiedzi merytorycznej.
2. **LegalRAG** — generuje odpowiedź z bazy; przy `RAG_JUDGE_ENABLED=true` uruchamia **sędziego źródeł** (reguły + LLM), do jednej poprawki.
3. **CalculatorAgent** / **DocumentGenerator** — obliczenia kadrowe i pisma PDF.

W czacie (SSE) zdarzenie `judge` informuje UI o wyniku weryfikacji.

## Dokumentacja

| Materiał | Opis |
|----------|------|
| [KodeksPracy_AI_Orchestration.pdf](docs/KodeksPracy_AI_Orchestration.pdf) | Prezentacja: architektura, orkiestracja agentów, RAG i kryteria wyróżnienia projektu GenAI |

## Wymagania

- Python 3.10+
- [Docker](https://www.docker.com/) (PostgreSQL z `docker-compose`)
- Klucze API: **Google** (Gemini) — obowiązkowe do czatu i kalkulatora; **Cohere** — do bazy wiedzy RAG (embeddingi, rerank)

## Szybki start

### 1. Klonowanie i konfiguracja

```bash
git clone https://github.com/matskpl/kodeks_pracy_ai.git
cd kodeks_pracy_ai
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/macOS
```

Uzupełnij w pliku `.env`:

```env
GOOGLE_API_KEY=twoj_klucz_google
COHERE_API_KEY=twoj_klucz_cohere
DATABASE_URL=postgresql://kodeks:kodeks@localhost:5432/kodekspracy
AUTH_SECRET=losowy-dlugi-sekret-produkcyjny
```

> `AUTH_SECRET` służy do hashowania haseł w bazie i podpisywania ciasteczka sesji — **nie commituj** prawdziwego `.env`.

### 2. PostgreSQL

```bash
docker compose up -d
python scripts/seed_postgres.py
```

Przy pierwszym uruchomieniu aplikacji schemat i seed tworzą się automatycznie, jeśli baza jest pusta. Opcjonalny import z `data/users.json` (jeśli plik istnieje).

### 3. Baza wiedzy (RAG)

Zatrzymaj aplikację, jeśli działa — Qdrant blokuje folder `data/qdrant` przy ingest w trakcie uvicorn.

```bash
python ingest.py --force
```

Pierwsze uruchomienie może pobrać teksty z ISAP/PIP/ZUS (cache w `data/scraped_chunks.json` po scrapingu).

### 4. Uruchomienie aplikacji

```bash
python main.py
```

Otwórz w przeglądarce:

- http://127.0.0.1:8000/login — logowanie  
- http://127.0.0.1:8000 — aplikacja (po zalogowaniu)

**Konta demonstracyjne:**

| Rola | Login | Hasło |
|------|--------|--------|
| Administrator kadr | `kadry` | `kadry123` |
| Pracownik (Jan Nowak) | `jnowak` | `jnowak123` |

Pozostali pracownicy: login jak w tabeli użytkowników, hasło `{login}123` (np. `akowalska123`).

Wylogowanie: przycisk **Wyloguj się** lub http://127.0.0.1:8000/logout

## Testy

```bash
pytest tests/ -q
```

Test integracji PostgreSQL (`tests/test_store_postgres.py`) wymaga działającej bazy z `DATABASE_URL`.

## Struktura repozytorium

```
agents/          # LLM: RAG, kalkulator, supervisor, judge, prompty HR
auth/            # PostgreSQL, sesje, hasła, profile pracowników
retrieval/       # BM25 + hybryda RRF
scrapers/        # ISAP, PIP, ZUS → chunki
services/        # Wypowiedzenia, urlop na żądanie, PDF
templates/       # UI (login, panel HR)
static/          # CSS, JS (podgląd PDF)
main.py          # FastAPI + SSE czat
ingest.py        # Indeksowanie Qdrant + BM25
docker-compose.yml
scripts/seed_postgres.py
```

## Dane w PostgreSQL

- **`employees`** — profil kadrowy (urlop, staż, nadgodziny, …)
- **`users`** — login, **hash hasła** (PBKDF2), rola (`employee` / `admin`), powiązanie z pracownikiem

Hasła nie są przechowywane jawnym tekstem.

## Zmienne środowiskowe (wybór)

| Zmienna | Opis |
|---------|------|
| `GOOGLE_API_KEY` | Gemini — agenci |
| `COHERE_API_KEY` | Embeddingi i rerank |
| `DATABASE_URL` | PostgreSQL |
| `AUTH_SECRET` | Hash haseł + podpis sesji |
| `RAG_HYBRID_ENABLED` | BM25 + dense (domyślnie `true`) |
| `RAG_JUDGE_ENABLED` | Pętla sędziego w LegalRAG (domyślnie `true`) |
| `RAG_JUDGE_MAX_REVISIONS` | Liczba poprawek po odrzuceniu przez sędziego (domyślnie `1`) |
| `RAG_JUDGE_MIN_SCORE` | Próg akceptacji grounding_score (domyślnie `0.75`) |
| `RAG_CHUNK_MAX_LEN` / `RAG_CHUNK_OVERLAP` | Parametry chunkowania (domyślnie 1500 / 250) |

Pełny szablon: `.env.example`.

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---------|-------------|
| Błąd połączenia z Postgres | `docker compose up -d`, sprawdź `DATABASE_URL` |
| Ingest: folder qdrant zablokowany | Zatrzymaj `python main.py`, potem `python ingest.py --force` |
| Wylogowanie wraca na stronę główną | Wyczyść ciasteczko `kp_session` lub wejdź na `/logout` |
| Brak odpowiedzi RAG | `COHERE_API_KEY` + ukończony ingest |

## Licencja i zastrzeżenia

Projekt demonstracyjny / wewnętrzny HR. Odpowiedzi AI nie zastępują porady prawnej — weryfikuj przepisy i indywidualne przypadki z kadrą lub radcą prawnym.

## Autor

Repozytorium: [matskpl/kodeks_pracy_ai](https://github.com/matskpl/kodeks_pracy_ai)
