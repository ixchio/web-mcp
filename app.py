import os
import time
import re
import html
import asyncio
import ipaddress
import socket
import json
from typing import Optional, Dict, Any, List, Tuple
import fnmatch
from urllib.parse import urlsplit
from datetime import datetime, timezone

import httpx
import trafilatura
import gradio as gr
from dateutil import parser as dateparser
from limits import parse
from limits.aio.storage import MemoryStorage
from limits.aio.strategies import MovingWindowRateLimiter

from analytics import record_request, last_n_days_count_df
from config import API_AUTH_TOKEN, RERANK_TOP_K, RAG_TOP_K

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
SERPER_NEWS_ENDPOINT = "https://google.serper.dev/news"
HEADERS = {"X-API-KEY": SERPER_API_KEY or "", "Content-Type": "application/json"}

# HTTP clients with connection pooling
SERPER_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
WEB_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)

SERPER_LIMITS = httpx.Limits(
    max_keepalive_connections=int(os.getenv("SERPER_KEEPALIVE", "32")),
    max_connections=int(os.getenv("SERPER_MAX_CONNECTIONS", "128")),
)
WEB_LIMITS = httpx.Limits(
    max_keepalive_connections=int(os.getenv("WEB_KEEPALIVE", "128")),
    max_connections=int(os.getenv("WEB_MAX_CONNECTIONS", "512")),
)

serper_client = httpx.AsyncClient(
    timeout=SERPER_TIMEOUT,
    limits=SERPER_LIMITS,
    http2=True,
    headers=HEADERS,
)

DEFAULT_USER_AGENT = os.getenv(
    "FETCH_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)

web_client = httpx.AsyncClient(
    timeout=WEB_TIMEOUT,
    limits=WEB_LIMITS,
    http2=True,
    follow_redirects=True,
    headers={"User-Agent": DEFAULT_USER_AGENT},
)

# Rate limiting (shared by both tools, process-local)
GLOBAL_RATE = parse(os.getenv("GLOBAL_RATE", "3000/minute"))
PER_IP_RATE = parse(os.getenv("PER_IP_RATE", "60/minute"))
storage = MemoryStorage()
limiter = MovingWindowRateLimiter(storage)

# Concurrency controls & resource caps
FETCH_MAX_BYTES = max(1024, int(os.getenv("FETCH_MAX_BYTES", "1500000")))
FETCH_CONCURRENCY = max(1, int(os.getenv("FETCH_CONCURRENCY", "64")))
SEARCH_CONCURRENCY = max(1, int(os.getenv("SEARCH_CONCURRENCY", "64")))
EXTRACT_CONCURRENCY = max(
    1,
    int(
        os.getenv(
            "EXTRACT_CONCURRENCY",
            str(max(4, (os.cpu_count() or 2) * 2)),
        )
    ),
)

SEARCH_CACHE_TTL = max(0, int(os.getenv("SEARCH_CACHE_TTL", "30")))
FETCH_CACHE_TTL = max(0, int(os.getenv("FETCH_CACHE_TTL", "300")))

# Controls for private/local address handling in fetch()
def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean-like env vars such as 1/true/yes/on."""
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}

# When True, allow any destination (disables SSRF guard — not recommended)
FETCH_ALLOW_PRIVATE = _env_flag("FETCH_ALLOW_PRIVATE", False)

# Optional comma/space separated host patterns to allow even if private, e.g.:
#   FETCH_PRIVATE_ALLOWLIST="*.internal.example.com, my-proxy.local"
FETCH_PRIVATE_ALLOWLIST = [
    p for p in re.split(r"[\s,]+", os.getenv("FETCH_PRIVATE_ALLOWLIST", "").strip()) if p
]

_search_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
_fetch_cache: Dict[str, Dict[str, Any]] = {}
_search_cache_lock: Optional[asyncio.Lock] = None
_fetch_cache_lock: Optional[asyncio.Lock] = None
_search_sema: Optional[asyncio.Semaphore] = None
_fetch_sema: Optional[asyncio.Semaphore] = None
_extract_sema: Optional[asyncio.Semaphore] = None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _get_cache_lock(name: str) -> asyncio.Lock:
    global _search_cache_lock, _fetch_cache_lock
    if name == "search":
        if _search_cache_lock is None:
            _search_cache_lock = asyncio.Lock()
        return _search_cache_lock
    if name == "fetch":
        if _fetch_cache_lock is None:
            _fetch_cache_lock = asyncio.Lock()
        return _fetch_cache_lock
    raise ValueError(f"Unknown cache lock: {name}")


def _get_semaphore(name: str) -> asyncio.Semaphore:
    global _search_sema, _fetch_sema, _extract_sema
    if name == "search":
        if _search_sema is None:
            _search_sema = asyncio.Semaphore(SEARCH_CONCURRENCY)
        return _search_sema
    if name == "fetch":
        if _fetch_sema is None:
            _fetch_sema = asyncio.Semaphore(FETCH_CONCURRENCY)
        return _fetch_sema
    if name == "extract":
        if _extract_sema is None:
            _extract_sema = asyncio.Semaphore(EXTRACT_CONCURRENCY)
        return _extract_sema
    raise ValueError(f"Unknown semaphore: {name}")


async def _cache_get(name: str, cache: Dict[Any, Any], key: Any):
    lock = _get_cache_lock(name)
    async with lock:
        entry = cache.get(key)
        if not entry:
            return None
        if time.time() > entry["expires_at"]:
            cache.pop(key, None)
            return None
        return entry["value"]


async def _cache_set(name: str, cache: Dict[Any, Any], key: Any, value: Any, ttl: int):
    if ttl <= 0:
        return
    lock = _get_cache_lock(name)
    async with lock:
        cache[key] = {"expires_at": time.time() + ttl, "value": value}


def _client_ip(request: Optional[gr.Request]) -> str:
    try:
        if request is None:
            return "unknown"
        headers = getattr(request, "headers", None) or {}
        xff = headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        client = getattr(request, "client", None)
        if client and getattr(client, "host", None):
            return client.host
    except Exception:
        pass
    return "unknown"


def _host_matches_allowlist(host: str) -> bool:
    """Return True if host matches any pattern in FETCH_PRIVATE_ALLOWLIST."""
    if not host:
        return False
    for pat in FETCH_PRIVATE_ALLOWLIST:
        # Support bare host equality and fnmatch-style patterns (*.foo.bar)
        if host == pat or fnmatch.fnmatch(host, pat):
            return True
    return False


async def _resolve_addresses(host: str) -> List[str]:
    def _resolve() -> List[str]:
        try:
            return list({ai[4][0] for ai in socket.getaddrinfo(host, None)})
        except Exception:
            return []

    return await asyncio.to_thread(_resolve)


async def _host_is_public(host: str) -> Tuple[bool, List[str]]:
    """Return (is_public, resolved_addresses).

    - If resolution fails, treat as public and let HTTP request decide.
    - Honors allowlist/env flags via the caller.
    """
    if not host:
        return False, []

    addresses = await _resolve_addresses(host)
    if not addresses:
        return True, []

    for addr in addresses:
        ip_obj = ipaddress.ip_address(addr)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_reserved
            or ip_obj.is_unspecified
        ):
            return False, addresses
    return True, addresses


async def _check_rate_limits(bucket: str, ip: str) -> Optional[str]:
    if not await limiter.hit(GLOBAL_RATE, "global"):
        return f"Global rate limit exceeded. Limit: {GLOBAL_RATE}."
    if ip != "unknown":
        if not await limiter.hit(PER_IP_RATE, f"{bucket}:{ip}"):
            return f"Per-IP rate limit exceeded. Limit: {PER_IP_RATE}."
    return None


def _domain_from_url(url: str) -> str:
    try:
        netloc = urlsplit(url).netloc
        return netloc.replace("www.", "")
    except Exception:
        return ""


def _iso_date_or_unknown(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    try:
        return dateparser.parse(date_str, fuzzy=True).strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_title_from_html(html_text: str) -> Optional[str]:
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    title = re.sub(r"\s+", " ", m.group(1)).strip()
    return html.unescape(title) if title else None


# ──────────────────────────────────────────────────────────────────────────────
# Tool: search (metadata only)
# ──────────────────────────────────────────────────────────────────────────────
async def search(
    query: str,
    search_type: str = "search",
    num_results: Optional[int] = 4,
    request: Optional[gr.Request] = None,
) -> Dict[str, Any]:
    """Perform a web or news search via Serper and return metadata only."""
    start_time = time.time()

    if not query or not query.strip():
        await record_request("search")
        return {"error": "Missing 'query'. Please provide a search query string."}

    query = query.strip()
    if num_results is None:
        num_results = 4
    try:
        num_results = max(1, min(20, int(num_results)))
    except (TypeError, ValueError):
        num_results = 4

    if search_type not in ["search", "news"]:
        search_type = "search"

    if not SERPER_API_KEY:
        await record_request("search")
        return {
            "error": "SERPER_API_KEY is not set. Export SERPER_API_KEY and try again."
        }

    ip = _client_ip(request)

    try:
        rl_message = await _check_rate_limits("search", ip)
        if rl_message:
            await record_request("search")
            return {"error": rl_message}

        cache_key = (query, search_type, num_results)
        cached = await _cache_get("search", _search_cache, cache_key)
        if cached:
            await record_request("search")
            return cached

        endpoint = (
            SERPER_NEWS_ENDPOINT if search_type == "news" else SERPER_SEARCH_ENDPOINT
        )
        payload: Dict[str, Any] = {"q": query, "num": num_results}
        if search_type == "news":
            payload["type"] = "news"
            payload["page"] = 1

        semaphore = _get_semaphore("search")
        await semaphore.acquire()
        try:
            resp = await serper_client.post(endpoint, json=payload)
        finally:
            semaphore.release()

        if resp.status_code != 200:
            await record_request("search")
            return {
                "error": f"Search API returned status {resp.status_code}. Check your API key and query."
            }

        data = resp.json()
        raw_results: List[Dict[str, Any]] = (
            data.get("news", []) if search_type == "news" else data.get("organic", [])
        )

        formatted: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_results[:num_results], start=1):
            entry = {
                "position": idx,
                "title": item.get("title"),
                "link": item.get("link"),
                "domain": _domain_from_url(item.get("link", "")),
                "snippet": item.get("snippet") or item.get("description"),
            }
            if search_type == "news":
                entry["source"] = item.get("source")
                entry["date"] = _iso_date_or_unknown(item.get("date"))
            formatted.append(entry)

        result = {
            "query": query,
            "search_type": search_type,
            "count": len(formatted),
            "results": formatted,
            "duration_s": round(time.time() - start_time, 2),
        }

        if not formatted:
            result["message"] = f"No {search_type} results found."

        await _cache_set("search", _search_cache, cache_key, result, SEARCH_CACHE_TTL)
        await record_request("search")

        return result

    except Exception as e:
        await record_request("search")
        return {"error": f"Search failed: {str(e)}"}


# ──────────────────────────────────────────────────────────────────────────────
# Tool: fetch (single URL fetch + extraction)
# ──────────────────────────────────────────────────────────────────────────────
async def fetch(
    url: str,
    timeout: int = 20,
    request: Optional[gr.Request] = None,
) -> Dict[str, Any]:
    """Fetch a single URL and extract the main readable content."""
    start_time = time.time()

    if not url or not isinstance(url, str):
        await record_request("fetch")
        return {"error": "Missing 'url'. Please provide a valid URL string."}
    if not url.lower().startswith(("http://", "https://")):
        await record_request("fetch")
        return {"error": "URL must start with http:// or https://."}

    try:
        timeout = max(5, min(60, int(timeout)))
    except (TypeError, ValueError):
        timeout = 20

    ip = _client_ip(request)

    try:
        host = urlsplit(url).hostname or ""
        if not host:
            await record_request("fetch")
            return {"error": "Invalid URL; unable to determine host."}
        rl_message = await _check_rate_limits("fetch", ip)
        if rl_message:
            await record_request("fetch")
            return {"error": rl_message}

        cache_key = (url, timeout)
        cached = await _cache_get("fetch", _fetch_cache, cache_key)
        if cached:
            await record_request("fetch")
            return cached

        is_public, addrs = await _host_is_public(host)
        if not is_public and not (FETCH_ALLOW_PRIVATE or _host_matches_allowlist(host)):
            await record_request("fetch")
            detail = f" (resolved: {', '.join(addrs)})" if addrs else ""
            return {
                "error": "Refusing to fetch private or local addresses." + detail,
                "host": host,
            }

        fetch_sema = _get_semaphore("fetch")
        await fetch_sema.acquire()
        try:
            async with web_client.stream("GET", url, timeout=timeout) as resp:
                status_code = resp.status_code
                total = 0
                chunks: List[bytes] = []
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > FETCH_MAX_BYTES:
                        break
                    chunks.append(chunk)
                body = b"".join(chunks)
                final_url_str = str(resp.url)
                encoding = resp.encoding or "utf-8"
        finally:
            fetch_sema.release()

        truncated = total > FETCH_MAX_BYTES
        # Extra guard: if final URL host ended up private due to a redirect and
        # the user hasn't allowed private hosts, refuse to return body content.
        try:
            final_host = urlsplit(final_url_str).hostname or ""
        except Exception:
            final_host = ""
        if final_host and not (FETCH_ALLOW_PRIVATE or _host_matches_allowlist(final_host)):
            final_public, _ = await _host_is_public(final_host)
            if not final_public:
                await record_request("fetch")
                return {
                    "error": "Refusing to fetch private or local addresses after redirect.",
                    "host": final_host,
                }
        text = body.decode(encoding, errors="ignore")

        extract_sema = _get_semaphore("extract")
        await extract_sema.acquire()
        try:
            content = await asyncio.to_thread(
                trafilatura.extract,
                text,
                include_formatting=False,
                include_comments=False,
            )
        finally:
            extract_sema.release()

        content = (content or "").strip()
        title = _extract_title_from_html(text) or ""
        domain = _domain_from_url(final_url_str)
        word_count = len(content.split()) if content else 0

        result = {
            "url": url,
            "final_url": final_url_str,
            "domain": domain,
            "status_code": status_code,
            "title": title,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "word_count": word_count,
            "content": content,
            "duration_s": round(time.time() - start_time, 2),
        }

        if truncated:
            result["truncated"] = True

        await _cache_set("fetch", _fetch_cache, cache_key, result, FETCH_CACHE_TTL)
        await record_request("fetch")
        return result

    except httpx.HTTPError as e:
        await record_request("fetch")
        return {"error": f"Network error while fetching: {str(e)}"}
    except Exception as e:
        await record_request("fetch")
        return {"error": f"Unexpected error while fetching: {str(e)}"}


# ──────────────────────────────────────────────────────────────────────────────
# Auth helper
# ──────────────────────────────────────────────────────────────────────────────
def _check_auth(request: Optional[gr.Request]) -> Optional[Dict[str, str]]:
    """If API_AUTH_TOKEN is set, validate the Authorization header.
    Returns an error dict on failure, or None on success."""
    if not API_AUTH_TOKEN:
        return None  # auth disabled
    if request is None:
        return None  # Gradio UI calls don't carry gr.Request in all modes
    headers = getattr(request, "headers", None) or {}
    auth = headers.get("authorization", "")
    if auth == f"Bearer {API_AUTH_TOKEN}":
        return None
    return {"error": "Unauthorized. Provide a valid Bearer token in the Authorization header."}


# ──────────────────────────────────────────────────────────────────────────────
# Tool: Ask (RAG Pipeline) — streaming
# ──────────────────────────────────────────────────────────────────────────────
rag_pipeline = None
reranker_instance = None


async def ask_rag(
    query: str,
    search_type: str = "search",
    num_results: int = 4,
    request: Optional[gr.Request] = None,
):
    """Search → fetch → chunk → embed → rerank → stream answer.
    Yields (partial_answer, sources_json_str) tuples for Gradio streaming."""
    global rag_pipeline, reranker_instance

    empty_sources = "[]"

    if not query or not query.strip():
        await record_request("search")
        yield "❌ Missing query.", empty_sources
        return

    auth_err = _check_auth(request)
    if auth_err:
        yield f"❌ {auth_err['error']}", empty_sources
        return

    # Lazy init models
    if rag_pipeline is None:
        yield "⏳ Loading ML models (first request only)...", empty_sources
        try:
            from rag import RAGPipeline
            from reranker import CrossEncoderReranker

            def _init_models():
                global rag_pipeline, reranker_instance
                if rag_pipeline is None:
                    rag_pipeline = RAGPipeline()
                    reranker_instance = CrossEncoderReranker()

            await asyncio.to_thread(_init_models)
        except Exception as e:
            yield f"❌ Failed to load ML models: {e}", empty_sources
            return

    yield "🔍 Searching the web...", empty_sources

    search_res = await search(query, search_type, num_results, request)
    if "error" in search_res:
        yield f"❌ {search_res['error']}", empty_sources
        return

    results = search_res.get("results", [])
    if not results:
        yield "❌ No search results found.", empty_sources
        return

    urls = [r["link"] for r in results if r.get("link")]
    url_to_title = {r["link"]: r.get("title", "") for r in results if r.get("link")}

    yield f"📥 Fetching {len(urls)} pages...", empty_sources

    fetch_tasks = [fetch(url, timeout=15, request=request) for url in urls]
    fetched_results = await asyncio.gather(*fetch_tasks)

    documents = []
    for f_res in fetched_results:
        if "content" in f_res and f_res["content"]:
            documents.append({"url": f_res["url"], "content": f_res["content"]})

    if not documents:
        yield "❌ Could not extract text from any result.", empty_sources
        return

    yield "🧠 Embedding & reranking...", empty_sources

    # Build index, retrieve, rerank (CPU-bound → thread)
    def _prepare_context():
        rag_pipeline.build_index(documents)
        retrieved = rag_pipeline.retrieve(query, top_k=RAG_TOP_K)
        if not retrieved:
            return None, None, []

        texts = [r["text"] for r in retrieved]
        reranked = reranker_instance.rerank(query, texts, top_k=RERANK_TOP_K)
        best_context = "\n\n".join([c[0] for c in reranked])

        sources = []
        for c_text, c_score in reranked:
            src = {"text": c_text[:200] + "..." if len(c_text) > 200 else c_text,
                   "reranker_score": round(float(c_score), 4)}
            for r in retrieved:
                if r["text"] == c_text:
                    src["url"] = r["source"]
                    src["title"] = url_to_title.get(r["source"], "")
                    break
            sources.append(src)
        return best_context, retrieved, sources

    try:
        best_context, retrieved, sources = await asyncio.to_thread(_prepare_context)
    except Exception as e:
        yield f"❌ ML Error: {e}", empty_sources
        return

    if best_context is None:
        yield "❌ No relevant context retrieved.", empty_sources
        return

    sources_json = json.dumps(sources, indent=2)

    # Stream the generation
    yield "✍️ Generating answer...", sources_json

    partial_answer = ""
    try:
        def _stream_gen():
            return list(rag_pipeline.generate_answer_stream(query, best_context))

        tokens = await asyncio.to_thread(_stream_gen)
        for tok in tokens:
            partial_answer += tok
            yield partial_answer, sources_json
    except Exception as e:
        yield f"❌ Generation error: {e}", sources_json
        return

    await record_request("search")


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Web MCP Server") as demo:
    gr.HTML(
        """
        <div style="background-color: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 16px; text-align: center;">
            <p style="color: rgb(59, 130, 246); margin: 0; font-size: 14px; font-weight: 500;">
                🤝 Community resource — please use responsibly to keep this service available for everyone
            </p>
        </div>
        """
    )

    gr.Markdown("# 🔍 Web Search MCP Server")
    gr.Markdown(
        "This server provides two composable MCP tools: **search** (metadata only) and **fetch** (single-URL extraction)."
    )

    with gr.Tabs():
        with gr.Tab("Ask (RAG)"):
            gr.Markdown("## 🧠 Search + AI Answer (Streaming)")
            rag_query_input = gr.Textbox(
                label="Question",
                placeholder='e.g. "What is the newest context window for GPT-4?"',
            )
            rag_run_button = gr.Button("Ask", variant="primary")
            rag_answer_output = gr.Textbox(
                label="AI Answer",
                lines=10,
                interactive=False,
            )
            rag_sources_output = gr.Textbox(
                label="Sources (JSON)",
                lines=6,
                interactive=False,
            )

            rag_run_button.click(
                fn=ask_rag,
                inputs=[rag_query_input],
                outputs=[rag_answer_output, rag_sources_output],
                api_name="ask_rag",
            )

            gr.Examples(
                examples=[
                    ["What is the speed of light?"],
                    ["Who won the last Super Bowl?"],
                    ["Explain the Model Context Protocol."],
                ],
                inputs=[rag_query_input],
                outputs=[rag_answer_output, rag_sources_output],
                fn=ask_rag,
                cache_examples=False,
            )

        with gr.Tab("Tools"):
            with gr.Row():
                # ── Search panel ───────────────────────────────────────────────
                with gr.Column(scale=3):
                    gr.Markdown("## Search (metadata only)")
                    query_input = gr.Textbox(
                        label="Search Query",
                        placeholder='e.g. "OpenAI news", "climate change 2024", "React hooks useState"',
                        info="Required",
                    )
                    search_type_input = gr.Radio(
                        choices=["search", "news"],
                        value="search",
                        label="Search Type",
                        info="Choose general web search or news",
                    )
                    num_results_input = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=4,
                        step=1,
                        label="Number of Results",
                        info="Optional (default 4)",
                    )
                    search_button = gr.Button("Run Search", variant="primary")
                    search_output = gr.JSON(
                        label="Search Results (metadata only)",
                    )

                    gr.Examples(
                        examples=[
                            ["OpenAI GPT-5 latest developments", "news", 5],
                            ["React hooks useState", "search", 4],
                            ["Apple Vision Pro reviews", "search", 4],
                            ["Tesla stock price today", "news", 6],
                        ],
                        inputs=[query_input, search_type_input, num_results_input],
                        outputs=search_output,
                        fn=search,
                        cache_examples=False,
                    )

                # ── Fetch panel ────────────────────────────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("## Fetch (single URL → extracted content)")
                    url_input = gr.Textbox(
                        label="URL",
                        placeholder="https://example.com/article",
                        info="Required: the URL to fetch and extract",
                    )
                    timeout_input = gr.Slider(
                        minimum=5,
                        maximum=60,
                        value=20,
                        step=1,
                        label="Timeout (seconds)",
                        info="Optional (default 20)",
                    )
                    fetch_button = gr.Button("Fetch & Extract", variant="primary")
                    fetch_output = gr.JSON(label="Fetched Content (structured)")

                    gr.Examples(
                        examples=[
                            ["https://news.ycombinator.com/"],
                            ["https://www.python.org/dev/peps/pep-0008/"],
                            ["https://en.wikipedia.org/wiki/Model_Context_Protocol"],
                        ],
                        inputs=[url_input],
                        outputs=fetch_output,
                        fn=fetch,
                        cache_examples=False,
                    )

            # Wire up buttons
            search_button.click(
                fn=search,
                inputs=[query_input, search_type_input, num_results_input],
                outputs=search_output,
                api_name=False,
            )
            fetch_button.click(
                fn=fetch,
                inputs=[url_input, timeout_input],
                outputs=fetch_output,
                api_name=False,
            )

        with gr.Tab("Analytics"):
            gr.Markdown("## Community Usage Analytics")
            gr.Markdown("Daily request counts (UTC), split by tool.")

            with gr.Row():
                with gr.Column():
                    search_plot = gr.BarPlot(
                        value=last_n_days_count_df("search", 14),
                        x="date",
                        y="count",
                        title="Daily Search Count",
                        tooltip=["date", "count", "full_date"],
                        height=350,
                        x_label_angle=-45,
                        container=False,
                    )
                with gr.Column():
                    fetch_plot = gr.BarPlot(
                        value=last_n_days_count_df("fetch", 14),
                        x="date",
                        y="count",
                        title="Daily Fetch Count",
                        tooltip=["date", "count", "full_date"],
                        height=350,
                        x_label_angle=-45,
                        container=False,
                    )

    # Refresh analytics on load
    demo.load(
        fn=lambda: (
            last_n_days_count_df("search", 14),
            last_n_days_count_df("fetch", 14),
        ),
        outputs=[search_plot, fetch_plot],
        api_name=False,
    )

    # Expose MCP tools
    gr.api(search, api_name="search")
    gr.api(fetch, api_name="fetch")


demo.queue(
    max_size=int(os.getenv("GRADIO_MAX_QUEUE", "256")),
    default_concurrency_limit=int(os.getenv("GRADIO_CONCURRENCY", "32")),
)


if __name__ == "__main__":
    # Launch with MCP server enabled
    demo.launch(mcp_server=True, show_api=True)
