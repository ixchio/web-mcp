# ─── analytics.py ──────────────────────────────────────────────────────────────
import os
import json
import asyncio
from datetime import datetime, timedelta, timezone
from filelock import FileLock  # pip install filelock
import pandas as pd  # already available in HF images

# Determine data directory based on environment
# 1. Check for environment variable override
# 2. Use /data if it exists and is writable (Hugging Face Spaces with persistent storage)
# 3. Use ./data for local development
DATA_DIR = os.getenv("ANALYTICS_DATA_DIR")
if not DATA_DIR:
    if os.path.exists("/data") and os.access("/data", os.W_OK):
        DATA_DIR = "/data"
        print("[Analytics] Using persistent storage at /data")
    else:
        DATA_DIR = "./data"
        print("[Analytics] Using local storage at ./data")

os.makedirs(DATA_DIR, exist_ok=True)

COUNTS_FILE = os.path.join(DATA_DIR, "request_counts.json")
LOCK_FILE = os.path.join(DATA_DIR, "analytics.lock")


# ──────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────────────
def _load_counts() -> dict:
    if not os.path.exists(COUNTS_FILE):
        return {}
    with open(COUNTS_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_counts(data: dict):
    with open(COUNTS_FILE, "w") as f:
        json.dump(data, f)


def _normalize_counts_schema(data: dict) -> dict:
    """
    Ensure data is {date: {"search": int, "fetch": int}}.
    Backward compatible with old schema {date: int}.
    """
    normalized = {}
    for day, value in data.items():
        if isinstance(value, dict):
            normalized[day] = {
                "search": int(value.get("search", 0)),
                "fetch": int(value.get("fetch", 0)),
            }
        else:
            # Old schema: total count as int → attribute to "search", keep fetch=0
            normalized[day] = {"search": int(value or 0), "fetch": 0}
    return normalized


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def _record_request_sync(tool: str) -> None:
    tool = (tool or "").strip().lower()
    if tool not in {"search", "fetch"}:
        # Ignore unknown tool buckets to keep charts clean
        tool = "search"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with FileLock(LOCK_FILE):
        data = _normalize_counts_schema(_load_counts())
        if today not in data:
            data[today] = {"search": 0, "fetch": 0}
        data[today][tool] = int(data[today].get(tool, 0)) + 1
        _save_counts(data)


async def record_request(tool: str) -> None:
    """Increment today's counter (UTC) for the given tool: 'search' or 'fetch'."""
    await asyncio.to_thread(_record_request_sync, tool)


def last_n_days_count_df(tool: str, n: int = 30) -> pd.DataFrame:
    """Return DataFrame with a row for each of the past n days for the given tool."""
    tool = (tool or "").strip().lower()
    if tool not in {"search", "fetch"}:
        tool = "search"

    now = datetime.now(timezone.utc)
    with FileLock(LOCK_FILE):
        data = _normalize_counts_schema(_load_counts())

    records = []
    for i in range(n):
        day = now - timedelta(days=n - 1 - i)
        day_key = day.strftime("%Y-%m-%d")
        display_date = day.strftime("%b %d")
        counts = data.get(day_key, {"search": 0, "fetch": 0})
        records.append(
            {
                "date": display_date,
                "count": int(counts.get(tool, 0)),
                "full_date": day_key,
            }
        )
    return pd.DataFrame(records)
