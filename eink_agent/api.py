"""FastAPI application for EInkAgent."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path as FilePath
import time
from typing import Annotated, Any, Literal

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path, Query
from fastapi.responses import FileResponse
import openai
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints, model_validator

from device_repository import compare_devices, get_device_detail, search_devices
from eink_agent.agent import MAX_HISTORY_MESSAGES, run_agent
from eink_agent.knowledge import KnowledgeBase, KnowledgeSearcher, embed_texts
from eink_agent.memory import MEMORY_DB_PATH, SessionMemory, SQLiteMemoryStore, update_summary

STATIC_DIR = FilePath(__file__).resolve().parent / "static"

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"

# The SDK retries timeouts, 429 and 5xx responses itself; 400 and 401 are
# not retried because sending the same request again cannot succeed.
MODEL_TIMEOUT_SECONDS = 60.0
MODEL_MAX_RETRIES = 2

MAX_MESSAGE_LENGTH = 1000
MAX_HISTORY_CONTENT_LENGTH = 4000
# The Agent keeps only the latest messages; this cap just bounds request size.
MAX_HISTORY_ITEMS = 20
# The page generates a UUID; anything else is rejected before it reaches SQL.
SESSION_ID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class DeviceComparisonRequest(BaseModel):
    """Validated request body for comparing devices."""

    device_ids: list[PositiveInt] = Field(min_length=2, max_length=5)


class HistoryMessage(BaseModel):
    """One earlier turn replayed by the client."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=MAX_HISTORY_CONTENT_LENGTH,
        ),
    ]


class AgentRequest(BaseModel):
    """A new user message plus the conversation so far."""

    model_config = ConfigDict(extra="forbid")

    message: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=MAX_MESSAGE_LENGTH,
        ),
    ]
    history: list[HistoryMessage] = Field(
        default_factory=list,
        max_length=MAX_HISTORY_ITEMS,
    )
    # With a session the server keeps the history, preferences and summary
    # (decision 009); without one it stores nothing, as before.
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)

    @model_validator(mode="after")
    def _one_source_of_history(self) -> "AgentRequest":
        if self.session_id and self.history:
            raise ValueError("send either session_id or history, not both")
        return self


class ToolCallView(BaseModel):
    """One tool call as shown to API clients."""

    name: str
    arguments: dict[str, Any] | None
    status: Literal["ok", "error"]
    error_type: str | None
    result_count: int
    result: Any


class AgentResponse(BaseModel):
    """The Agent's answer and how it was produced."""

    answer: str
    status: Literal["ok", "degraded", "blocked"]
    tool_calls: list[ToolCallView]
    latency_ms: int
    model: str
    # Preferences in effect after this turn; None without a session or when
    # memory could not be read.
    preferences: dict[str, Any] | None = None


class PreferenceEvent(BaseModel):
    field: str
    value: Any
    created_at: str


class SessionView(BaseModel):
    """A stored conversation, so the page can restore it after a refresh."""

    messages: list[HistoryMessage]
    preferences: dict[str, Any]
    preference_history: list[PreferenceEvent]


@dataclass(frozen=True)
class AgentRuntime:
    """The model client, model name and optional knowledge searcher."""

    client: Any
    model: str
    knowledge: Any = None


@lru_cache(maxsize=1)
def _deepseek_client(api_key: str) -> OpenAI:
    """Reuse one HTTP client across requests for the same key."""
    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def _knowledge_searcher() -> KnowledgeSearcher | None:
    """Knowledge search is on only when an embedding service is configured.

    Built once: the corpus and vector index are loaded at first use. A
    corpus that no longer matches the index raises here, at startup of the
    feature, instead of returning stale passages.
    """
    api_key = os.environ.get("EMBEDDING_API_KEY")
    base_url = os.environ.get("EMBEDDING_BASE_URL")
    if not api_key or not base_url:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=MODEL_TIMEOUT_SECONDS, max_retries=MODEL_MAX_RETRIES)
    model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    kb = KnowledgeBase.from_files()
    if kb.embedding_model != model:
        # Vectors from different models are not comparable: every score
        # would be meaningless without any error.
        raise ValueError(f"Knowledge index was built with {kb.embedding_model}, not {model}; rebuild it.")
    return KnowledgeSearcher(
        kb,
        embed=lambda text: embed_texts(client, model, [text])[0],
    )


def get_agent_runtime() -> AgentRuntime:
    """Build the model runtime from environment variables.

    Tests replace this dependency, so they never need an API key.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="服务端未配置 DEEPSEEK_API_KEY，请在启动服务的终端中设置后重启。",
        )

    return AgentRuntime(
        client=_deepseek_client(api_key),
        model=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        knowledge=_knowledge_searcher(),
    )


def get_memory_store() -> SQLiteMemoryStore:
    """Session memory store; tests replace this dependency."""
    return SQLiteMemoryStore(os.environ.get("MEMORY_DB_PATH") or MEMORY_DB_PATH)


app = FastAPI(title="EInkAgent", version="0.5.0")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the single-page chat interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str,str]:
    """Return the service health status"""
    return {"status": "ok"}


@app.post("/agent")
def ask_agent(
    request: AgentRequest,
    runtime: Annotated[AgentRuntime, Depends(get_agent_runtime)],
    store: Annotated[SQLiteMemoryStore, Depends(get_memory_store)],
    background_tasks: BackgroundTasks,
) -> AgentResponse:
    """Answer one message, with history from the client or from the session."""
    started = time.perf_counter()
    history = [message.model_dump() for message in request.history]
    memory = None
    if request.session_id:
        memory = SessionMemory(store, request.session_id)
        try:
            history = store.messages(request.session_id)[-MAX_HISTORY_MESSAGES:]
        except Exception:  # noqa: BLE001 - memory is optional (decision 009)
            history = []

    try:
        result = run_agent(
            request.message,
            client=runtime.client,
            model=runtime.model,
            history=history,
            knowledge=runtime.knowledge,
            memory=memory,
        )
    except openai.APITimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="模型响应超时，请稍后重试。",
        ) from error
    except (openai.APIError, RuntimeError) as error:
        # The SDK error may include the request URL or provider messages.
        raise HTTPException(
            status_code=502,
            detail="模型服务暂时不可用，请稍后重试。",
        ) from error

    preferences = None
    if memory is not None:
        preferences = _save_turn(memory, request.message, result.answer, store_turn=result.status == "ok")
        background_tasks.add_task(_summarize_quietly, store, memory.session_id, runtime)

    return AgentResponse(
        answer=result.answer,
        status=result.status,
        tool_calls=[
            ToolCallView(
                name=call["name"],
                arguments=call["arguments"],
                status=call["status"],
                error_type=call.get("error", {}).get("type") if call["status"] == "error" else None,
                result_count=call["result_count"],
                result=call["result"],
            )
            for call in result.tool_trace
        ],
        latency_ms=round((time.perf_counter() - started) * 1000),
        model=runtime.model,
        preferences=preferences,
    )


def _save_turn(memory: SessionMemory, message: str, answer: str, *, store_turn: bool) -> dict[str, Any] | None:
    """Store the turn and return current preferences; failures are dropped.

    Only completed turns are stored as context, as the page did before
    sessions. The user already has the answer, so a storage error must not
    turn it into an HTTP error.
    """
    try:
        if store_turn:
            memory.store.add_messages(
                memory.session_id,
                [{"role": "user", "content": message}, {"role": "assistant", "content": answer}],
            )
        return memory.preferences()
    except Exception:  # noqa: BLE001 - memory is optional (decision 009)
        return None


def _summarize_quietly(store: SQLiteMemoryStore, session_id: str, runtime: AgentRuntime) -> None:
    """Runs after the response is sent; update_summary keeps the old summary
    on model errors, and a storage error is dropped here."""
    try:
        update_summary(store, session_id, client=runtime.client, model=runtime.model, window=MAX_HISTORY_MESSAGES)
    except Exception:  # noqa: BLE001 - memory is optional (decision 009)
        pass


@app.get("/sessions/{session_id}")
def get_session(
    session_id: Annotated[str, Path(pattern=SESSION_ID_PATTERN)],
    store: Annotated[SQLiteMemoryStore, Depends(get_memory_store)],
) -> SessionView:
    """Messages and preferences of one session, to restore the page."""
    try:
        return SessionView(
            messages=store.messages(session_id),
            preferences=store.preferences(session_id),
            preference_history=store.preference_history(session_id),
        )
    except Exception as error:  # noqa: BLE001 - the details may include file paths
        raise HTTPException(status_code=503, detail="会话记忆暂时无法读取。") from error


@app.get("/devices")
def get_devices(
    brand: str | None = None,
    max_price: Annotated[int | None, Query(ge=0)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict[str, Any]]:
    """Search devices using HTTP query parameters."""
    return search_devices(
        brand=brand,
        max_price=max_price,
        limit=limit,
    )


@app.post("/devices/compare")
def compare_device_options(
    request: DeviceComparisonRequest,
) -> list[dict[str, Any]]:
    """Compare devices supplied in a JSON request body."""
    try:
        return compare_devices(request.device_ids)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@app.get("/devices/{device_id}")
def get_device(
    device_id: Annotated[int, Path(ge=1)],
) -> dict[str, Any]:
    """Return one device by its ID."""
    device = get_device_detail(device_id)

    if device is None:
        raise HTTPException(status_code=404, detail="Device not found")

    return device
