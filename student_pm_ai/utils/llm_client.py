"""
llm_client.py
--------------
Thin, provider-agnostic wrapper around an LLM API used by the Planning Agent.

Supports THREE modes, auto-detected from what's configured in .env:

    1. Gemini    - if GEMINI_API_KEY is set (free tier available via
                   Google AI Studio: https://aistudio.google.com)
    2. Anthropic - if ANTHROPIC_API_KEY is set (used if no Gemini key)
    3. Mock      - if neither key is set (fully offline, deterministic)

Design goal: the whole project must run out-of-the-box even with NO API
key and NO extra packages installed - so students can explore the
planning/risk/replanning LOOP immediately, then plug in a real LLM later
just by adding a key to .env. No code changes needed either way.

Set LLM_PROVIDER=gemini or LLM_PROVIDER=anthropic in .env to force a
specific provider instead of relying on auto-detection.
"""

import json
import os
import re
import time
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
def _clean_model_name(name: str) -> str:
    """Strip stray quotes/whitespace and a redundant leading 'models/'
    (a common copy-paste mistake, since Google's own error messages
    include that prefix) - the SDK adds the prefix itself."""
    name = name.strip().strip('"').strip("'")
    if name.startswith("models/"):
        name = name[len("models/"):]
    return name


GEMINI_MODEL = _clean_model_name(os.environ.get("GEMINI_MODEL", "gemini-flash-latest"))


def _forced_provider() -> Optional[str]:
    value = os.environ.get("LLM_PROVIDER", "").strip().lower()
    return value if value in ("gemini", "anthropic") else None


def _try_gemini_client():
    """Return a Gemini client if the SDK is installed and a key is set."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai  # noqa: local import so the package is optional
        return genai.Client(api_key=api_key)
    except ImportError:
        return None


def _try_anthropic_client():
    """Return an Anthropic client if the SDK is installed and a key is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # noqa: local import so the package is optional
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        return None


def active_provider() -> Optional[str]:
    """
    Return which provider will actually be used right now:
    "gemini", "anthropic", or None (mock mode).

    Gemini is checked first (it's the free option) unless LLM_PROVIDER
    forces a specific choice.
    """
    forced = _forced_provider()
    if forced == "gemini":
        return "gemini" if _try_gemini_client() else None
    if forced == "anthropic":
        return "anthropic" if _try_anthropic_client() else None

    if _try_gemini_client() is not None:
        return "gemini"
    if _try_anthropic_client() is not None:
        return "anthropic"
    return None


def llm_is_live() -> bool:
    """True if a real LLM will be used, False if running in mock mode."""
    return active_provider() is not None


def _extract_json(text: str) -> Dict[str, Any]:
    """LLMs sometimes wrap JSON in prose or code fences - pull the object out."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text[:500]}")
    return json.loads(match.group(0))


def _call_gemini(client, system_prompt: str, user_prompt: str) -> str:
    from google.genai import types  # noqa: local import, mirrors _try_gemini_client

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=2000,
        ),
    )
    return response.text


def _call_anthropic(client, system_prompt: str, user_prompt: str) -> str:
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def generate_json(system_prompt: str, user_prompt: str, mock_fn=None) -> Dict[str, Any]:
    """
    Call whichever LLM provider is configured and parse its reply as JSON.

    If no API key is configured (or the matching package isn't installed),
    `mock_fn()` is used instead so the rest of the pipeline (risk scoring,
    replanning, UI) keeps working with zero cost and zero setup.

    Transient server errors (e.g. Gemini/Anthropic returning a temporary
    503 "high demand" error) are retried automatically with backoff. If
    every retry fails, we fall back to `mock_fn()` (when available) rather
    than crashing the app - so a temporary provider outage never blocks a
    demo, it just silently produces a mock plan for that one call.
    """
    provider = active_provider()

    if provider is None:
        if mock_fn is None:
            raise RuntimeError(
                "No GEMINI_API_KEY or ANTHROPIC_API_KEY configured, and no "
                "mock_fn supplied. Set one of them in your .env file to use "
                "a real LLM (Gemini has a free tier - see "
                "https://aistudio.google.com)."
            )
        return mock_fn()

    max_retries = 3
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "gemini":
                client = _try_gemini_client()
                text = _call_gemini(client, system_prompt, user_prompt)
            else:  # "anthropic"
                client = _try_anthropic_client()
                text = _call_anthropic(client, system_prompt, user_prompt)
            return _extract_json(text)

        except Exception as e:  # noqa: broad on purpose - many SDK-specific error types
            last_error = e
            is_transient = _is_transient_error(e)
            if is_transient and attempt < max_retries:
                wait = 2 ** attempt  # 2s, 4s, 8s
                print(f"[llm_client] {provider} temporarily unavailable "
                      f"(attempt {attempt}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
                continue
            break

    # All retries exhausted (or a non-transient error occurred).
    if mock_fn is not None:
        print(f"[llm_client] {provider} call failed after retries, "
              f"falling back to offline mock plan. Last error: {last_error}")
        return mock_fn()

    raise last_error


def _is_transient_error(e: Exception) -> bool:
    """Heuristic: treat rate-limit/overload/server errors as retryable."""
    text = str(e).lower()
    class_name = type(e).__name__.lower()
    transient_markers = ("503", "overloaded", "unavailable", "high demand",
                          "rate limit", "429", "timeout", "servererror")
    return any(m in text or m in class_name for m in transient_markers)
