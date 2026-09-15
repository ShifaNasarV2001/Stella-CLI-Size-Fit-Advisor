"""The only place in the program that talks to the model.

Routed through OpenRouter's OpenAI-compatible endpoint rather than the
Anthropic API directly, because the credential available in this environment
is an OpenRouter key, not an Anthropic key. Everything else in the app calls
`LLMClient.call(...)` and gets back either plain text or a validated Pydantic
object — transport concerns (timeouts, retries, schema enforcement) are
handled here and collapse into a single `LLMUnavailable` the CLI knows how to
render.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Sequence, Type, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from . import tracing

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-opus-5"
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 800

# Reasoning tokens are billed and count against max_tokens, and on a short
# structured-extraction call they can crowd out the JSON itself (observed
# truncation at the default budget). "low" keeps extraction/response calls
# fast and cheap without materially changing output quality for this task.
REASONING_EFFORT = os.getenv("STELLA_REASONING_EFFORT", "low")


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or returned something unusable.

    Raised only after retries are exhausted. The CLI catches this and keeps the
    session alive rather than dropping the user's collected profile.
    """


def _strict_schema(model: Type[BaseModel]) -> dict:
    """Pydantic's JSON schema, tightened to satisfy OpenAI-style strict mode.

    Strict structured outputs require every object to set
    `additionalProperties: false` and list every property as required (fields
    with defaults still get a value — the model just always supplies one).
    Pydantic doesn't emit either by default, so this walks the schema and adds
    both to every object node, recursively into `$defs`.
    """

    def tighten(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for item in node:
                tighten(item)
        return node

    return tighten(model.model_json_schema())


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.model = model or os.getenv("STELLA_MODEL", DEFAULT_MODEL)
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMUnavailable(
                "No API key found. Copy .env.example to .env and set OPENROUTER_API_KEY."
            )
        # The SDK already retries connection errors, 408/409/429 and 5xx with
        # exponential backoff. Configuring it beats hand-rolling a retry loop
        # that would double-retry on top of it.
        self._client = openai.OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
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

        chat_messages = [{"role": "system", "content": system}, *messages]
        extra_body: dict[str, Any] = {"reasoning": {"effort": REASONING_EFFORT}}
        response_format = None
        if response_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": _strict_schema(response_schema),
                },
            }

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=chat_messages,
                response_format=response_format,
                extra_body=extra_body,
            )

            choice = response.choices[0]
            if choice.finish_reason == "length":
                raise LLMUnavailable(
                    f"Response was cut off at the {max_tokens}-token limit before completing."
                )

            content = (choice.message.content or "").strip()
            if not content:
                raise LLMUnavailable("Model returned an empty response.")

            if response_schema is not None:
                try:
                    result: str | T = response_schema.model_validate(json.loads(content))
                except (json.JSONDecodeError, ValidationError) as exc:
                    trace.error = "schema_validation"
                    raise LLMUnavailable(
                        f"Model output did not match {response_schema.__name__}: {exc}"
                    ) from exc
            else:
                result = content

            usage = getattr(response, "usage", None)
            if usage is not None:
                trace.input_tokens = getattr(usage, "prompt_tokens", None)
                trace.output_tokens = getattr(usage, "completion_tokens", None)

        except openai.RateLimitError as exc:
            trace.error = "rate_limit"
            trace.retry_count = self.max_retries
            raise LLMUnavailable(
                "Rate limited by the API after retries. Wait a moment and try again."
            ) from exc
        except openai.APITimeoutError as exc:
            trace.error = "timeout"
            trace.retry_count = self.max_retries
            raise LLMUnavailable(
                f"The model did not respond within {DEFAULT_TIMEOUT_SECONDS:.0f}s."
            ) from exc
        except openai.APIConnectionError as exc:
            trace.error = "connection"
            trace.retry_count = self.max_retries
            raise LLMUnavailable("Could not reach the API. Check your network connection.") from exc
        except openai.APIStatusError as exc:
            trace.error = f"http_{exc.status_code}"
            detail = ""
            try:
                detail = f": {exc.response.json().get('error', {}).get('message', '')}"
            except Exception:
                pass
            raise LLMUnavailable(f"API returned HTTP {exc.status_code}{detail}") from exc
        finally:
            trace.latency_ms = int((time.monotonic() - started) * 1000)
            tracing.record(trace)

        return result
