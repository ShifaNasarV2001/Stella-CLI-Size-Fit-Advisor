"""Per-call observability.

Off by default so a normal session stays quiet; set STELLA_TRACE=1 to get a
JSONL record of every model call next to the code. Useful for showing that the
token cost per turn is bounded even though full state is re-injected each time.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

TRACE_FILE = Path(os.getenv("STELLA_TRACE_FILE", ".stella_traces.jsonl"))


def enabled() -> bool:
    return os.getenv("STELLA_TRACE", "").strip() in {"1", "true", "yes", "on"}


@dataclass
class Trace:
    call_type: str
    model: str
    prompt_char_count: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int = 0
    retry_count: int = 0
    error: str | None = None
    timestamp: str = ""


def record(trace: Trace) -> None:
    """Append one trace. Never raises: observability must not break a session."""
    if not enabled():
        return
    trace.timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    try:
        with TRACE_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(trace)) + "\n")
    except OSError:
        pass
