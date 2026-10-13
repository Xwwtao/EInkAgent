"""FastAPI application for EInkAgent."""

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path as FilePath
import time
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.responses import FileResponse
import openai
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, StringConstraints

from device_repository import compare_devices, get_device_detail, search_devices
from eink_agent.agent import run_agent
from eink_agent.knowledge import KnowledgeBase, KnowledgeSearcher, embed_texts

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
) -> AgentResponse:
    """Answer one message, using the client-supplied history as context."""
    started = time.perf_counter()

    try:
        result = run_agent(
            request.message,
            client=runtime.client,
            model=runtime.model,
            history=[message.model_dump() for message in request.history],
            knowledge=runtime.knowledge,
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
    )


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
