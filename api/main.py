import time
import uuid
from typing import Dict

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from api.logging_config import configure_logging, request_log_payload, write_request_log
from api.models import AskRequest, AskResponse, AddDocumentRequest, AddDocumentResponse, WebSocketRequest
from guardrails.pii import mask_pii_text
from guardrails.prompt_injection import enforce_prompt_injection_guardrail

logger = configure_logging("logs/requests.jsonl")
app = FastAPI(title="Nykaa Domain Support Agent", version="1.0.0")
_ADDED_DOCUMENTS: Dict[str, str] = {}


def _trace_id():
    return str(uuid.uuid4())


def _log(trace_id, method, path, started, request_text, status_code):
    write_request_log(
        logger,
        request_log_payload(
            trace_id=trace_id,
            method=method,
            path=path,
            duration_ms=(time.perf_counter() - started) * 1000,
            request_text=request_text,
            status_code=status_code,
        ),
    )


@app.middleware("http")
async def structured_request_logging(request: Request, call_next):
    trace_id = request.headers.get("X-Trace-ID", _trace_id())
    started = time.perf_counter()
    body = await request.body()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(request.scope, receive)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Trace-ID"] = trace_id
        return response
    finally:
        _log(
            trace_id,
            request.method,
            request.url.path,
            started,
            body.decode("utf-8", errors="replace") if body else "",
            status_code,
        )


def _run_crew(query):
    # Preserve the existing Task 11 test seam. If agents.crew.run_crew has
    # been monkeypatched, call that test double directly. Otherwise use the
    # real integrated pipeline.
    from agents import crew as crew_module
    current = crew_module.run_crew
    original = getattr(crew_module, "_ORIGINAL_RUN_CREW", None)
    if original is not None and current is not original:
        return current(query)

    from integration.pipeline import run_support_request
    return run_support_request(query)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    trace_id = _trace_id()
    try:
        masked = mask_pii_text(request.query)
        enforce_prompt_injection_guardrail(masked)
        result = _run_crew(masked)
        return AskResponse(
            response=result.response,
            source_type=result.source_type,
            grounded=result.grounded,
            session_id=request.session_id,
            trace_id=trace_id,
        )
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"detail": str(exc), "trace_id": trace_id},
        )
    
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"{type(exc).__name__}: {exc}",
                "trace_id": trace_id,
            },
        )


@app.post("/add-document", response_model=AddDocumentResponse)
def add_document(request: AddDocumentRequest):
    trace_id = _trace_id()
    if request.document_id in _ADDED_DOCUMENTS:
        return JSONResponse(
            status_code=409,
            content={"detail": "document_id already exists", "trace_id": trace_id},
        )
    _ADDED_DOCUMENTS[request.document_id] = request.content
    return AddDocumentResponse(
        document_id=request.document_id,
        accepted=True,
        message="Document accepted into the API document registry.",
        trace_id=trace_id,
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = websocket.query_params.get("session_id", str(uuid.uuid4()))
    try:
        while True:
            raw = await websocket.receive_json()
            request = WebSocketRequest.model_validate(raw)
            trace_id, started, status_code = _trace_id(), time.perf_counter(), 200
            try:
                masked = mask_pii_text(request.query)
                enforce_prompt_injection_guardrail(masked)
                from agents.memory import ask_with_memory
                result = ask_with_memory(session_id, masked)
                await websocket.send_json({
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "response": result["crew_result"],
                    "cache_hit": result.get("cache_hit", False),
                    "review_approved": result.get("review_approved", False),
                })
            except ValueError as exc:
                status_code = 400
                await websocket.send_json({
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "error": str(exc),
                })
            except Exception:
                status_code = 500
                await websocket.send_json({
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "error": "Internal support-agent error.",
                })
            finally:
                _log(trace_id, "WEBSOCKET", "/ws/chat", started, request.query, status_code)
    except WebSocketDisconnect:
        return
