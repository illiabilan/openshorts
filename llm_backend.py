"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible", "openrouter", "fireworks", "groq", "custom"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    if explicit == "anthropic":
        return "anthropic"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    return (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")


def model_name() -> str:
    return (os.environ.get("LLM_MODEL") or "").strip() or DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    return httpx.Client(timeout=_timeout(), **kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost


PROVIDER_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}


def test_llm_connection(
    provider_name: Optional[str] = None,
    key: Optional[str] = None,
    model: Optional[str] = None,
    custom_base_url: Optional[str] = None,
) -> Tuple[bool, str]:
    """Test connectivity to the requested LLM provider without running a video job."""
    prov = (provider_name or provider() or "gemini").strip().lower()

    if prov == "gemini":
        gemini_key = (key or os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY") or "").strip()
        if not gemini_key or gemini_key == "your_gemini_api_key_here":
            return False, "API ключ для Google Gemini не вказано або він недійсний."
        target_model = (model or os.environ.get("GEMINI_MODEL") or os.environ.get("LLM_MODEL") or "gemini-3.6-flash").strip()
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            resp = client.models.generate_content(
                model=target_model,
                contents="Ping. Respond with OK",
            )
            reply = (resp.text or "").strip()
            return True, f"З'єднання з Gemini успішне! Модель: {target_model} (відповідь: {reply[:30]})"
        except Exception as e:
            return False, f"Помилка підключення до Gemini: {e}"

    if prov == "anthropic":
        ant_key = (key or os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not ant_key:
            return False, "API ключ для Anthropic не вказано."
        target_model = (model or "claude-3-5-haiku-20241022").strip()
        headers = {
            "x-api-key": ant_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": target_model,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Ping. Respond with OK"}],
        }
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post("https://api.anthropic.com/v1/messages", json=body, headers=headers)
                if res.status_code == 200:
                    return True, f"З'єднання з Anthropic успішне! Модель: {target_model}"
                return False, f"Anthropic повернув код {res.status_code}: {res.text[:200]}"
        except Exception as e:
            return False, f"Помилка підключення до Anthropic: {e}"

    # OpenAI-compatible providers: openrouter, fireworks, openai, groq, ollama, custom, etc.
    url = (custom_base_url or os.environ.get("LLM_BASE_URL") or PROVIDER_BASE_URLS.get(prov) or "").strip().rstrip("/")
    if not url:
        url = "https://api.openai.com/v1" if prov == "openai" else "http://localhost:11434/v1"

    api_key = (key or os.environ.get("LLM_API_KEY") or ("ollama" if prov == "ollama" else "")).strip()
    if prov != "ollama" and not api_key:
        return False, f"API ключ для {prov} не вказано."

    target_model = (model or os.environ.get("LLM_MODEL") or ("llama3.1:8b" if prov == "ollama" else "gpt-4o-mini")).strip()
    headers = {"Authorization": f"Bearer {api_key or 'ollama'}", "Content-Type": "application/json"}
    if prov == "openrouter":
        headers["HTTP-Referer"] = "https://openshorts.local"
        headers["X-Title"] = "OpenShorts"

    body = {
        "model": target_model,
        "messages": [{"role": "user", "content": "Ping. Respond with OK"}],
        "max_tokens": 15,
        "temperature": 0.1,
    }
    try:
        with httpx.Client(timeout=15.0) as client:
            res = client.post(f"{url}/chat/completions", json=body, headers=headers)
            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices") or []
                reply = choices[0].get("message", {}).get("content", "").strip() if choices else "OK"
                return True, f"З'єднання з {prov} успішне! Модель {target_model} відповіла: {reply[:40]}"
            return False, f"{prov} повернув код {res.status_code}: {res.text[:200]}"
    except httpx.ConnectError:
        return False, f"Не вдалося з'єднатися з сервером {url}. Перевірте чи він запущений."
    except Exception as e:
        return False, f"Помилка підключення до {prov}: {e}"
