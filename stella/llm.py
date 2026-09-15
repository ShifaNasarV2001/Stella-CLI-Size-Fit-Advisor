"""The only place in the program that talks to the Claude API.

Everything else calls `LLMClient.call(...)` and gets back either plain text or
a validated Pydantic object. Transport concerns — timeouts, retries, rate
limits — are handled here and collapse into a single `LLMUnavailable` that the
CLI knows how to render.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence, Type, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from . import tracing

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 8192


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or returned something unusable.

    Raised only after retries are exhausted. The CLI catches this and keeps the
    session alive rather than dropping the user's collected profile.
    """


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model = model or os.getenv("STELLA_MODEL", DEFAULT_MODEL)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        # The SDK already implements exponential backoff over connection errors,
        # 408/409/429 and 5xx. Configuring it beats hand-rolling a retry loop
        # that would double-retry on top of it.
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self.max_retries = max_retries

    def call(
        self,
        system: str,
        messages: Sequence[dict[str, Any]],
        response_schema: Type[T] | None = None,
        *,
        call_type: str = "generic",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str | T:
        """One model call. Returns validated `response_schema` when given, else text."""
        started = time.monotonic()
        trace = tracing.Trace(
            call_type=call_type,
            model=self.model,
            prompt_char_count=len(system) + sum(len(str(m.get("content", ""))) for m in messages),
        )

        try:
            if response_schema is not None:
                response = self._client.messages.parse(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=list(messages),
                    output_format=response_schema,
                )
                parsed = response.parsed_output
                if parsed is None:
                    raise LLMUnavailable(
                        f"Model returned no parsable {response_schema.__name__}."
                    )
                result: str | T = parsed
            else:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=list(messages),
                )
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()
                if not text:
                    raise LLMUnavailable("Model returned an empty response.")
                result = text

            usage = getattr(response, "usage", None)
            if usage is not None:
                trace.input_tokens = getattr(usage, "input_tokens", None)
                trace.output_tokens = getattr(usage, "output_tokens", None)

        except anthropic.RateLimitError as exc:
            trace.error = "rate_limit"
            trace.retry_count = self.max_retries
            raise LLMUnavailable(
                "Rate limited by the API after retries. Wait a moment and try again."
            ) from exc
        except anthropic.APITimeoutError as exc:
            trace.error = "timeout"
            trace.retry_count = self.max_retries
            raise LLMUnavailable(
                f"The model did not respond within {DEFAULT_TIMEOUT_SECONDS:.0f}s."
            ) from exc
        except anthropic.APIConnectionError as exc:
            trace.error = "connection"
            trace.retry_count = self.max_retries
            raise LLMUnavailable("Could not reach the API. Check your network connection.") from exc
        except anthropic.APIStatusError as exc:
            trace.error = f"http_{exc.status_code}"
            raise LLMUnavailable(f"API returned HTTP {exc.status_code}.") from exc
        except ValidationError as exc:
            trace.error = "schema_validation"
            raise LLMUnavailable(
                f"Model output did not match {response_schema.__name__ if response_schema else '?'}: "
                f"{exc.error_count()} field error(s)."
            ) from exc
        finally:
            trace.latency_ms = int((time.monotonic() - started) * 1000)
            tracing.record(trace)

        return result
