"""Serwer FastAPI — KodeksPracy AI (MetalTech Sp. z o.o.)."""

from __future__ import annotations

import base64
import json
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import Cookie, Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic_ai import Agent
from sse_starlette.sse import EventSourceResponse

from agents.calculator import (
    build_calculator_deps,
    build_calculator_prompt,
    calculator_agent,
)
from agents.doc_gen import (
    build_document_deps,
    build_document_prompt,
    document_generator_agent,
)
from agents.legal_rag import (
    build_legal_rag_deps,
    build_rag_user_prompt,
    chunks_to_sources,
    legal_rag_agent,
    retrieve_legal_chunks,
    run_legal_rag,
    split_text_for_sse,
)
from agents.rag_grounding import grounding_disclaimer
from agents.models import (
    CalculateRequest,
    DocumentGenerateApiResponse,
    DocumentGenerateOutput,
    DocumentGenerateRequest,
    KalkulatorOutput,
)
from auth.access import (
    assert_chat_scope,
    assert_document_allowed,
    build_user_context,
    is_admin,
    require_employee_access,
    resolve_target_employee_id,
)
from auth.deps import (
    SESSION_COOKIE,
    apply_session_cookie,
    clear_session_cookie,
    get_current_user,
    require_admin,
)
from auth.employee_calc import employee_to_calculator_input, finalize_calculator_output
from auth.employee_chat import profile_snapshot_answer_for_chat
from auth.models import (
    AuthUser,
    EmployeeDetailResponse,
    EmployeePublic,
    LoginRequest,
    MeResponse,
    UserRole,
)
from auth.password import verify_password
from auth.session import create_session_token, load_session_token
from auth.store import find_user, get_employee, list_employees
from agents.supervisor import (
    AGENT_LABELS,
    AgentIntent,
    build_general_prompt,
    calculator_input_from_message,
    classify_intent,
    document_request_from_message,
    general_agent,
    supervisor_agent,
)
from config import BASE_DIR, TEMPLATES_DIR, get_settings
from scrapers.pipeline import load_cache_metadata, load_cached_chunks, scrape_all
from services.document_pdf import build_document_pdf
from services.leave import (
    InsufficientOnDemandLeaveError,
    InsufficientVacationHoursError,
    OnDemandLeaveRequest,
    OnDemandLeaveResult,
    process_on_demand_leave_request,
)
from services.leave.on_demand import pools_from_employee_snapshot
from vector_store import (
    collection_point_count,
    get_qdrant_client,
    knowledge_base_is_ready,
    upsert_chunks,
)

INGESTED = False


def configure_logging() -> None:
    settings = get_settings()
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, colorize=True)


def ensure_knowledge_base() -> None:
    global INGESTED
    if INGESTED:
        return
    settings = get_settings()
    if not settings.cohere_api_key:
        logger.warning("Brak COHERE_API_KEY — pomijam ingest bazy wiedzy.")
        return

    client = get_qdrant_client()
    cache_meta = load_cache_metadata()
    scraped_at = cache_meta.get("scraped_at")
    chunks = load_cached_chunks()

    if knowledge_base_is_ready(
        client=client,
        expected_chunks=len(chunks) if chunks else None,
        scraped_at=scraped_at,
    ):
        INGESTED = True
        logger.info(
            "Baza wiedzy Qdrant gotowa — {} punktów (bez ponownego embedowania).",
            collection_point_count(client),
        )
        return

    if not chunks:
        logger.info("Brak cache — uruchamiam scraping ISAP/PIP/ZUS...")
        chunks = scrape_all(use_cache=False)
        cache_meta = load_cache_metadata()
        scraped_at = cache_meta.get("scraped_at")

    if settings.max_ingest_chunks > 0:
        chunks = chunks[: settings.max_ingest_chunks]

    if not chunks:
        logger.error("Brak chunków do ingestu — uruchom: python ingest.py --refresh")
        return

    logger.info("Pierwszy ingest bazy wiedzy Qdrant ({} chunków)...", len(chunks))
    upsert_chunks(chunks, client=client, recreate=True, scraped_at=scraped_at)
    INGESTED = True
    logger.success("Baza wiedzy Qdrant gotowa (ISAP + PIP + ZUS)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    from auth.store import ensure_database

    ensure_database()
    settings = get_settings()
    if not settings.google_api_key:
        logger.warning("Brak GOOGLE_API_KEY — agenci LLM nie będą działać poprawnie.")
    if not settings.cohere_api_key:
        logger.warning("Brak COHERE_API_KEY — embeddingi i rerank nie będą działać.")
    else:
        ensure_knowledge_base()
    yield
    logger.info("Zamykanie KodeksPracy AI")


app = FastAPI(
    title="KodeksPracy AI",
    description="Multi-Agent System dla MetalTech Sp. z o.o.",
    version="1.0.0",
    lifespan=lifespan,
)

_static = BASE_DIR / "static"
if _static.is_dir():
    app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/login", response_class=HTMLResponse)
async def login_page() -> HTMLResponse:
    html_path = TEMPLATES_DIR / "login.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Brak pliku templates/login.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/", response_model=None)
async def index(
    kp_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    settings = get_settings()
    if kp_session:
        user = load_session_token(
            kp_session,
            secret=settings.auth_secret,
            max_age_sec=settings.session_max_age_sec,
        )
        if user:
            html_path = TEMPLATES_DIR / "index.html"
            if not html_path.exists():
                raise HTTPException(status_code=500, detail="Brak pliku templates/index.html")
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return RedirectResponse(url="/login", status_code=302)


@app.post("/api/auth/login")
async def auth_login(payload: LoginRequest, response: Response) -> MeResponse:
    account = find_user(payload.username)
    settings = get_settings()
    if not account or not verify_password(payload.password, account.password_hash, secret=settings.auth_secret):
        raise HTTPException(status_code=401, detail="Nieprawidłowy login lub hasło.")

    user = AuthUser(
        username=account.username,
        role=account.role,
        display_name=account.display_name,
        employee_id=account.employee_id,
    )
    token = create_session_token(user, secret=settings.auth_secret)
    apply_session_cookie(response, token, max_age_sec=settings.session_max_age_sec)
    employee = get_employee(user.employee_id) if user.employee_id else None
    return MeResponse(user=user, employee=employee)


@app.post("/api/auth/logout")
async def auth_logout() -> JSONResponse:
    """Wylogowanie — Set-Cookie musi być na zwracanej odpowiedzi (nie na osobnym Response)."""
    resp = JSONResponse({"status": "ok"})
    clear_session_cookie(resp)
    return resp


@app.get("/logout")
async def logout_page() -> RedirectResponse:
    """Wylogowanie z przekierowaniem (działa też bez JS)."""
    resp = RedirectResponse(url="/login", status_code=302)
    clear_session_cookie(resp)
    return resp


@app.get("/api/auth/me", response_model=MeResponse)
async def auth_me(user: AuthUser = Depends(get_current_user)) -> MeResponse:
    employee = get_employee(user.employee_id) if user.employee_id else None
    return MeResponse(user=user, employee=employee)


@app.get("/api/employees", response_model=list[EmployeePublic])
async def employees_list(_admin: AuthUser = Depends(require_admin)) -> list[EmployeePublic]:
    return [
        EmployeePublic(
            id=e.id,
            imie_nazwisko=e.imie_nazwisko,
            stanowisko=e.stanowisko,
            dzial=e.dzial,
            urlop_pozostaly=e.urlop_pozostaly,
            wymiar_etatu=e.wymiar_etatu,
        )
        for e in list_employees()
    ]


@app.get("/api/employees/{employee_id}", response_model=EmployeeDetailResponse)
async def employee_detail(
    employee_id: str,
    user: AuthUser = Depends(get_current_user),
) -> EmployeeDetailResponse:
    require_employee_access(user, employee_id)
    emp = get_employee(employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Nie znaleziono pracownika.")
    return EmployeeDetailResponse(employee=emp)


async def _stream_agent_text(agent: Agent, prompt: str, deps) -> AsyncIterator[str]:
    kwargs = {"deps": deps} if deps is not None else {}
    async with agent.run_stream(prompt, **kwargs) as run:
        async for text in run.stream_text(delta=True):
            if text:
                yield text


async def _stream_supervisor_general(message: str) -> AsyncIterator[str]:
    prompt = build_general_prompt(message)
    async for token in _stream_agent_text(general_agent, prompt, None):
        yield token


async def _stream_legal_rag_events(
    rag_query: str,
    settings,
) -> AsyncIterator[dict[str, str]]:
    """
    SSE dla LegalRAG: ze sędzią — generacja + weryfikacja + ewentualna poprawka, potem stream;
    bez sędziego — klasyczny stream tokenów z modelu.
    """
    if settings.rag_judge_enabled:
        yield {
            "event": "judge",
            "data": json.dumps({"status": "checking"}, ensure_ascii=False),
        }
        text, chunks, meta = await run_legal_rag(rag_query)
        sources = chunks_to_sources(chunks)
        yield {
            "event": "sources",
            "data": json.dumps(
                {"sources": [s.model_dump() for s in sources]},
                ensure_ascii=False,
            ),
        }
        yield {
            "event": "judge",
            "data": json.dumps(
                {
                    "status": "done",
                    "accepted": meta.final_accepted,
                    "score": meta.final_score,
                    "revisions": meta.revision_attempts,
                    "issues": meta.judge_issues[:5],
                },
                ensure_ascii=False,
            ),
        }
        for part in split_text_for_sse(text):
            yield {
                "event": "token",
                "data": json.dumps({"text": part}, ensure_ascii=False),
            }
        return

    chunks = retrieve_legal_chunks(rag_query)
    deps = build_legal_rag_deps(user_query=rag_query, retrieved_chunks=chunks)
    rag_prompt = build_rag_user_prompt(rag_query, chunks)
    sources = chunks_to_sources(chunks)
    yield {
        "event": "sources",
        "data": json.dumps(
            {"sources": [s.model_dump() for s in sources]},
            ensure_ascii=False,
        ),
    }
    full_answer: list[str] = []
    async with legal_rag_agent.run_stream(rag_prompt, deps=deps) as run:
        async for token in run.stream_text(delta=True):
            if token:
                full_answer.append(token)
                yield {
                    "event": "token",
                    "data": json.dumps({"text": token}, ensure_ascii=False),
                }
    note = grounding_disclaimer("".join(full_answer), chunks)
    if note:
        yield {
            "event": "token",
            "data": json.dumps({"text": note}, ensure_ascii=False),
        }


async def chat_event_generator(message: str, user: AuthUser) -> AsyncIterator[dict[str, str]]:
    if not message.strip():
        yield {"event": "error", "data": json.dumps({"detail": "Pusta wiadomość."}, ensure_ascii=False)}
        return

    settings = get_settings()
    if not settings.google_api_key:
        yield {
            "event": "error",
            "data": json.dumps(
                {"detail": "Skonfiguruj GOOGLE_API_KEY w pliku .env"},
                ensure_ascii=False,
            ),
        }
        return

    try:
        employees = list_employees()
        assert_chat_scope(user, message, employees)
        own = get_employee(user.employee_id) if user.employee_id else None
        context_prefix = build_user_context(user, own)
        scoped_message = f"{context_prefix}\n\nPytanie użytkownika: {message}"

        profile_reply = profile_snapshot_answer_for_chat(message, user, employees, own)
        if profile_reply:
            yield {
                "event": "meta",
                "data": json.dumps(
                    {
                        "agent": "profile",
                        "agent_label": "Profil pracownika — dane kadrowe",
                        "confidence": 1.0,
                        "reasoning": "Dane kadrowe z profilu pracownika — bez ogólnego asystenta LLM.",
                    },
                    ensure_ascii=False,
                ),
            }
            yield {
                "event": "token",
                "data": json.dumps({"text": profile_reply}, ensure_ascii=False),
            }
            yield {
                "event": "sources",
                "data": json.dumps({"sources": []}, ensure_ascii=False),
            }
            yield {"event": "done", "data": json.dumps({"status": "ok"}, ensure_ascii=False)}
            return

        route = await classify_intent(scoped_message, raw_user_message=message)
        intent = route.intent
        query = route.refined_query or scoped_message

        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "agent": intent.value,
                    "agent_label": AGENT_LABELS[intent],
                    "confidence": route.confidence,
                    "reasoning": route.reasoning,
                },
                ensure_ascii=False,
            ),
        }

        from agents.routing_rules import is_explicit_calculator_request

        if intent == AgentIntent.LEGAL_RAG:
            calc_fallback = calculator_input_from_message(message)
            if calc_fallback and is_explicit_calculator_request(message):
                intent = AgentIntent.CALCULATOR
            else:
                if settings.cohere_api_key:
                    ensure_knowledge_base()
                rag_query = message
                async for ev in _stream_legal_rag_events(rag_query, settings):
                    yield ev

        if intent == AgentIntent.CALCULATOR:
            profile_calc = profile_snapshot_answer_for_chat(message, user, employees, own)
            if profile_calc:
                yield {
                    "event": "token",
                    "data": json.dumps({"text": profile_calc}, ensure_ascii=False),
                }
            else:
                calc_input = calculator_input_from_message(message)
                if calc_input is None:
                    if settings.cohere_api_key:
                        ensure_knowledge_base()
                    async for ev in _stream_legal_rag_events(message, settings):
                        yield ev
                elif own and not is_admin(user) and not is_explicit_calculator_request(message):
                    calc_input = employee_to_calculator_input(
                        own,
                        typ_obliczenia=calc_input.typ_obliczenia,
                        dodatkowe_info=message,
                    )
                    deps = build_calculator_deps()
                    prompt = build_calculator_prompt(calc_input)
                    result = await calculator_agent.run(prompt, deps=deps)
                    output = finalize_calculator_output(
                        result.output,
                        calc_input.typ_obliczenia,
                        own,
                    )
                    lines = [output.wyliczenie_opis]
                    if output.wynik_etykieta and output.wynik_glowny is not None:
                        u = output.wynik_jednostka or ""
                        lines.insert(
                            0,
                            f"{output.wynik_etykieta}: {output.wynik_glowny:g} {u}".strip(),
                        )
                    elif calc_input.typ_obliczenia == "urlop_wypoczynkowy":
                        lines.insert(0, f"Urlop wypoczynkowy: {output.urlop_dni} dni")
                    lines.insert(0, f"Podstawa prawna: {', '.join(output.podstawa_prawna)}")
                    text = "\n".join(lines)
                    yield {
                        "event": "token",
                        "data": json.dumps({"text": text}, ensure_ascii=False),
                    }
                else:
                    deps = build_calculator_deps()
                    prompt = build_calculator_prompt(calc_input)
                    result = await calculator_agent.run(prompt, deps=deps)
                    output: KalkulatorOutput = result.output
                    lines = [f"{output.wyliczenie_opis}"]
                    if output.wynik_etykieta and output.wynik_glowny is not None:
                        u = output.wynik_jednostka or ""
                        lines.insert(
                            0,
                            f"{output.wynik_etykieta}: {output.wynik_glowny:g} {u}".strip(),
                        )
                    elif calc_input.typ_obliczenia == "urlop_wypoczynkowy":
                        lines.insert(0, f"Urlop wypoczynkowy: {output.urlop_dni} dni")
                    elif calc_input.typ_obliczenia == "wypowiedzenie_umowy":
                        lines.insert(
                            0,
                            f"Okres wypowiedzenia: {output.wypowiedzenie_miesiace} mies.",
                        )
                    elif calc_input.typ_obliczenia == "nadgodziny":
                        lines.insert(0, "Nadgodziny (wyliczenie): patrz opis")
                    else:
                        lines.insert(0, f"Urlop: {output.urlop_dni} dni")
                    lines.insert(0, f"Podstawa prawna: {', '.join(output.podstawa_prawna)}")
                    text = "\n".join(lines)
                    yield {
                        "event": "token",
                        "data": json.dumps({"text": text}, ensure_ascii=False),
                    }

        elif intent == AgentIntent.DOCUMENT:
            doc_request = document_request_from_message(query)
            deps = build_document_deps()
            prompt = build_document_prompt(doc_request)
            result = await document_generator_agent.run(prompt, deps=deps)
            doc: DocumentGenerateOutput = result.output
            text = f"{doc.tytul}\n\n{doc.tresc}\n\nPodstawy prawne: {', '.join(doc.podstawy_prawne)}"
            yield {
                "event": "token",
                "data": json.dumps({"text": text}, ensure_ascii=False),
            }

        else:
            async for token in _stream_supervisor_general(scoped_message):
                yield {
                    "event": "token",
                    "data": json.dumps({"text": token}, ensure_ascii=False),
                }

        if intent != AgentIntent.LEGAL_RAG:
            yield {
                "event": "sources",
                "data": json.dumps({"sources": []}, ensure_ascii=False),
            }

        yield {"event": "done", "data": json.dumps({"status": "ok"}, ensure_ascii=False)}

    except HTTPException as exc:
        yield {
            "event": "error",
            "data": json.dumps({"detail": exc.detail}, ensure_ascii=False),
        }
    except Exception as exc:
        logger.exception("Błąd streamingu czatu: {}", exc)
        yield {
            "event": "error",
            "data": json.dumps({"detail": str(exc)}, ensure_ascii=False),
        }


@app.get("/api/stream-chat")
async def stream_chat(
    message: str = Query(..., min_length=1, max_length=4000),
    user: AuthUser = Depends(get_current_user),
):
    return EventSourceResponse(chat_event_generator(message, user))


@app.post("/api/calculate", response_model=KalkulatorOutput)
async def calculate(
    payload: CalculateRequest,
    user: AuthUser = Depends(get_current_user),
) -> KalkulatorOutput:
    settings = get_settings()
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Skonfiguruj GOOGLE_API_KEY w pliku .env")

    if user.role == UserRole.ADMIN:
        target_id = resolve_target_employee_id(user, payload.employee_id)
        emp = get_employee(target_id)
        if not emp:
            raise HTTPException(status_code=404, detail="Nie znaleziono pracownika.")
        calc_input = employee_to_calculator_input(
            emp,
            typ_obliczenia=payload.typ_obliczenia,
            dodatkowe_info=payload.dodatkowe_info,
            liczba_nadgodzin=payload.liczba_nadgodzin,
        )
    else:
        emp = get_employee(user.employee_id or "")
        if not emp:
            raise HTTPException(status_code=403, detail="Brak profilu pracownika.")
        calc_input = employee_to_calculator_input(
            emp,
            typ_obliczenia=payload.typ_obliczenia,
            dodatkowe_info=payload.dodatkowe_info,
            liczba_nadgodzin=payload.liczba_nadgodzin,
        )

    deps = build_calculator_deps()
    prompt = build_calculator_prompt(calc_input)
    logger.info("POST /api/calculate — {} typ: {}", user.username, calc_input.typ_obliczenia)
    try:
        result = await calculator_agent.run(prompt, deps=deps)
        return finalize_calculator_output(
            result.output,
            calc_input.typ_obliczenia,
            emp,
        )
    except Exception as exc:
        logger.exception("Błąd kalkulatora: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/leave/on-demand", response_model=OnDemandLeaveResult)
async def apply_on_demand_leave(
    payload: OnDemandLeaveRequest,
    user: AuthUser = Depends(get_current_user),
) -> OnDemandLeaveResult:
    """
    Przetwarza urlop na żądanie: −1 dzień z puli 4 dni/rok, −scheduled_hours z puli godzinowej.
    """
    if user.role == UserRole.ADMIN:
        employee_id = resolve_target_employee_id(user, payload.employee_id)
    else:
        if payload.employee_id and payload.employee_id != user.employee_id:
            raise HTTPException(
                status_code=403,
                detail="Możesz złożyć wniosek urlopu na żądanie tylko dla siebie.",
            )
        employee_id = user.employee_id or ""
    require_employee_access(user, employee_id)
    emp = get_employee(employee_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Nie znaleziono pracownika.")

    if payload.employee_id != employee_id:
        payload = payload.model_copy(update={"employee_id": employee_id})

    pools = pools_from_employee_snapshot(
        on_demand_used_days=emp.urlop_na_zadanie_wykorzystany,
        urlop_roczny_dni=emp.urlop_roczny_dni,
        urlop_wykorzystany_dni=emp.urlop_wykorzystany,
        wymiar_etatu=emp.wymiar_etatu,
    )
    try:
        result = process_on_demand_leave_request(pools, payload)
    except (InsufficientOnDemandLeaveError, InsufficientVacationHoursError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "POST /api/leave/on-demand — {} dni wniosku, pula na żądanie po: {}",
        len(payload.days),
        result.pools_after.on_demand_pool_days,
    )
    return result


@app.post("/api/document", response_model=DocumentGenerateApiResponse)
async def generate_document(
    payload: DocumentGenerateRequest,
    user: AuthUser = Depends(get_current_user),
) -> DocumentGenerateApiResponse:
    settings = get_settings()
    if not settings.google_api_key:
        raise HTTPException(status_code=503, detail="Skonfiguruj GOOGLE_API_KEY w pliku .env")

    assert_document_allowed(user, payload.typ_pisma)

    if user.role == UserRole.ADMIN:
        target_id = resolve_target_employee_id(user, payload.employee_id)
    else:
        target_id = user.employee_id or ""
    require_employee_access(user, target_id)
    emp = get_employee(target_id)
    if not emp:
        raise HTTPException(status_code=404, detail="Nie znaleziono pracownika.")

    if user.role != UserRole.ADMIN:
        if payload.imie_nazwisko.strip().casefold() != emp.imie_nazwisko.casefold():
            raise HTTPException(status_code=403, detail="Możesz generować pisma tylko na swoje dane.")
    else:
        payload = payload.model_copy(
            update={
                "imie_nazwisko": emp.imie_nazwisko,
                "stanowisko": emp.stanowisko,
            }
        )

    deps = build_document_deps()
    prompt = build_document_prompt(payload)
    logger.info("POST /api/document — {} / {} typ: {}", user.username, emp.id, payload.typ_pisma)
    try:
        result = await document_generator_agent.run(prompt, deps=deps)
        doc = result.output
        pdf_bytes = build_document_pdf(doc, settings.company)
        return DocumentGenerateApiResponse(
            **doc.model_dump(),
            pdf_base64=base64.b64encode(pdf_bytes).decode("ascii"),
        )
    except FileNotFoundError as exc:
        logger.error("Brak czcionek PDF: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Błąd generatora pism: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/health")
async def health() -> JSONResponse:
    settings = get_settings()
    client = get_qdrant_client()
    return JSONResponse(
        {
            "status": "ok",
            "company": settings.company.nazwa,
            "knowledge_base": INGESTED or collection_point_count(client) > 0,
            "qdrant_points": collection_point_count(client),
            "google_api_key_set": bool(settings.google_api_key),
            "cohere_api_key_set": bool(settings.cohere_api_key),
        }
    )


if __name__ == "__main__":
    configure_logging()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
