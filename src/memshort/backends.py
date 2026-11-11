"""The model calls `memshort om` makes: one completion, a system prompt and one user message in, text and usage out.

A model is named `<endpoint>/<model-id>`. The endpoint is either `pi`, which hands the call to the pi CLI and so
reaches every provider and login pi knows (`pi/deepseek/deepseek-flash`, `pi/anthropic/claude-haiku-4-5`), or an
HTTP endpoint memshort calls itself: one of `BUILTINS`, or one the user defines in the selection's `endpoints`.
An HTTP endpoint speaks one of two APIs: `openai` (chat completions, which most servers and gateways offer) or
`anthropic` (the Messages API). There are no tools and no conversation, so the request carries memshort's prompt
and nothing else.

This module never reads a credential from anywhere but the environment variable an endpoint names, never prints
one, and never includes a request or response body in an error beyond the provider's own short error message.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from http.client import HTTPException
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

APIS = ("openai", "anthropic")
PI = "pi"
ANTHROPIC_VERSION = "2023-06-01"
# The Messages API requires an output limit; an observer reply is a fraction of this.
DEFAULT_MAX_TOKENS = 16_000
# A provider's error message is shown up to this length; anything longer is cut.
ERROR_CHARS = 300
RESPONSE_LIMIT = 16 << 20
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Everything after the endpoint: one argument with no whitespace or control characters, not an option.
_ID = re.compile(r"^[^\s\x00-\x1f\x7f-][^\s\x00-\x1f\x7f]*$")
# pi's own `provider/model`: the provider is one path segment, the model may carry more slashes and `:` versions.
_PI_MODEL = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._-]*/[^\s\x00-\x1f\x7f]+$")


class BackendError(ValueError):
    """A model that cannot be named, an endpoint that cannot be used, or a call that failed. Never holds a secret."""


def url(value: object) -> str:
    """An HTTP(S) base URL without credentials, query or fragment, and without a trailing slash."""
    if not isinstance(value, str) or any(ord(c) < 33 or ord(c) == 127 for c in value):
        raise BackendError("base_url must be an HTTP(S) URL without whitespace")
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme in ("http", "https")
            and bool(parsed.hostname)
            and parsed.port != 0
            and not (
                parsed.username or parsed.password or parsed.query or parsed.fragment
            )
        )
    except ValueError:
        valid = False
    if not valid:
        raise BackendError(
            "base_url must be HTTP(S), with no credentials, query or fragment"
        )
    return value.rstrip("/")


def endpoint_name(name: object) -> str:
    if not isinstance(name, str) or not _NAME.match(name) or name == PI:
        raise BackendError(
            f"endpoint name must be lowercase letters, digits, - and _, and not {PI!r}; got {name!r}"
        )
    return name


@dataclass(frozen=True)
class Endpoint:
    """An HTTP endpoint memshort calls directly.

    `api_key_env` lists the environment variables tried in order; the first one set is the key. None of them set is
    no key, which a local server accepts. `extra_body` is merged into every request body, for the switches a provider
    needs, such as turning thinking off. `max_tokens` is the output limit; on `openai` it is only sent when set.
    """

    api: str
    base_url: str
    api_key_env: tuple[str, ...] = ()
    max_tokens: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    # A hosted API refuses a call without a key, so the call fails before it is sent; a local server needs none.
    key_required: bool = False

    def __post_init__(self) -> None:
        if self.api not in APIS:
            raise BackendError(
                f"api must be one of {', '.join(APIS)}, not {self.api!r}"
            )
        object.__setattr__(self, "base_url", url(self.base_url))
        keys = (
            (self.api_key_env,)
            if isinstance(self.api_key_env, str)
            else self.api_key_env
        )
        if not isinstance(keys, (tuple, list)) or not all(
            isinstance(k, str) and _ENV.match(k) for k in keys
        ):
            raise BackendError(
                "api_key_env must name environment variables, never hold a key"
            )
        object.__setattr__(self, "api_key_env", tuple(keys))
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise BackendError(
                f"max_tokens must be a positive whole number, not {self.max_tokens!r}"
            )
        if not isinstance(self.extra_body, dict):
            raise BackendError(
                f"extra_body must be a JSON object, not {self.extra_body!r}"
            )
        reserved = sorted(
            set(self.extra_body) & {"model", "messages", "system", "stream"}
        )
        if reserved:
            raise BackendError(f"extra_body cannot set {', '.join(reserved)}")

    def base(self) -> str:
        return self.base_url

    def key(self) -> str | None:
        """The first key set among `api_key_env`; a required key that is missing is an error naming the variables."""
        for variable in self.api_key_env:
            value = os.environ.get(variable, "").strip()
            if value:
                return value
        if self.key_required:
            raise BackendError(f"no API key: set {' or '.join(self.api_key_env)}")
        return None

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape of a user endpoint in the selection file."""
        out: dict[str, Any] = {"api": self.api, "base_url": self.base_url}
        if self.api_key_env:
            out["api_key_env"] = self.api_key_env[0]
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        if self.extra_body:
            out["extra_body"] = self.extra_body
        return out

    def describe(self) -> str:
        key = (
            f", key from {' or '.join(self.api_key_env)}"
            if self.api_key_env
            else ", no key"
        )
        key += "" if self.key_required or not self.api_key_env else " if set"
        extra = (
            f", extra_body {json.dumps(self.extra_body, sort_keys=True)}"
            if self.extra_body
            else ""
        )
        limit = f", max_tokens {self.max_tokens}" if self.max_tokens is not None else ""
        return f"{self.api} API at {self.base()}{key}{limit}{extra}"


def _builtin(
    api: str, base: str, name: str, standard: str | None, **kw: Any
) -> Endpoint:
    # MEMSHORT_<NAME>_API_KEY comes first, so a key for compaction need not sit in a variable the harness itself
    # reads: Claude Code bills ANTHROPIC_API_KEY instead of a subscription when it is set. A built-in's URL is fixed;
    # OPENAI_BASE_URL and the like often belong to the harness, so a gateway is an endpoint of its own.
    keys = (f"MEMSHORT_{name.upper()}_API_KEY", *((standard,) if standard else ()))
    return Endpoint(api, base, keys, key_required=standard is not None, **kw)


BUILTINS: dict[str, Endpoint] = {
    "anthropic": _builtin(
        "anthropic", "https://api.anthropic.com", "anthropic", "ANTHROPIC_API_KEY"
    ),
    # DeepSeek's V4 models think by default; observations need none of it.
    "deepseek": _builtin(
        "openai",
        "https://api.deepseek.com",
        "deepseek",
        "DEEPSEEK_API_KEY",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    "ollama": _builtin("openai", "http://localhost:11434/v1", "ollama", None),
    "openai": _builtin(
        "openai", "https://api.openai.com/v1", "openai", "OPENAI_API_KEY"
    ),
    "openrouter": _builtin(
        "openai", "https://openrouter.ai/api/v1", "openrouter", "OPENROUTER_API_KEY"
    ),
}


def endpoint_of(name: str, data: object) -> Endpoint:
    """A user endpoint from its JSON shape; every problem names the endpoint."""
    try:
        endpoint_name(name)
        if not isinstance(data, dict):
            raise BackendError("must be an object")
        unknown = sorted(
            set(data) - {"api", "base_url", "api_key_env", "max_tokens", "extra_body"}
        )
        if unknown:
            raise BackendError(f"unknown keys {', '.join(unknown)}")
        if "api" not in data or "base_url" not in data:
            raise BackendError("needs api and base_url")
        key = data.get("api_key_env")
        if key is not None and not isinstance(key, str):
            raise BackendError(
                "api_key_env must be the name of one environment variable"
            )
        return Endpoint(
            data["api"],
            data["base_url"],
            (key,) if key else (),
            data.get("max_tokens"),
            data.get("extra_body") or {},
            key_required=bool(key),
        )
    except BackendError as err:
        raise BackendError(f"endpoint {name}: {err}") from None


@dataclass(frozen=True)
class Route:
    """Where one model's calls go: through pi (`endpoint` None) or to an HTTP endpoint."""

    model: str
    backend: str
    model_id: str
    endpoint: Endpoint | None


def route(model: object, endpoints: dict[str, Endpoint] | None = None) -> Route:
    """The route `model` names. A user endpoint with a built-in's name replaces the built-in."""
    if not isinstance(model, str) or "/" not in model.strip():
        raise BackendError(
            f"model must look like <endpoint>/<model-id>, e.g. anthropic/claude-haiku-4-5, not {model!r}"
        )
    model = model.strip()
    backend, _, model_id = model.partition("/")
    if backend == PI:
        if not _PI_MODEL.match(model_id):
            raise BackendError(
                f"a pi model must look like pi/<provider>/<model>, e.g. pi/deepseek/deepseek-flash, not {model!r}"
            )
        return Route(model, backend, model_id, None)
    known = {**BUILTINS, **(endpoints or {})}
    if backend not in known:
        raise BackendError(
            f"unknown endpoint {backend!r} in {model!r}; use {PI}/<provider>/<model>, one of {', '.join(sorted(known))}, "
            "or add one with `memshort endpoint add`"
        )
    if not _ID.match(model_id):
        raise BackendError(f"model id in {model!r} must be one word without whitespace")
    return Route(model, backend, model_id, known[backend])


def zero_usage() -> dict:
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def complete(r: Route, system: str, prompt: str, timeout: float) -> tuple[str, dict]:
    """One completion: the reply's text and its usage in pi's shape. A direct call costs 0 in that record, since
    memshort keeps no price list; the token counts are the provider's."""
    if r.endpoint is None:
        return _pi(r.model_id, system, prompt, timeout)
    if r.endpoint.api == "anthropic":
        return _anthropic(r, system, prompt, timeout)
    return _openai(r, system, prompt, timeout)


class _NoRedirect(HTTPRedirectHandler):
    """A redirect could carry the key to another host, so none is followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(
    endpoint: Endpoint, path: str, payload: dict | None, timeout: float
) -> dict:
    """One JSON request. Proxies come from the environment as usual; redirects are refused."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    key = endpoint.key()
    if endpoint.api == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if key:
            headers["x-api-key"] = key
    elif key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        endpoint.base() + path,
        data=body,
        headers=headers,
        method="POST" if body else "GET",
    )
    try:
        with build_opener(_NoRedirect()).open(req, timeout=timeout) as response:
            raw = response.read(RESPONSE_LIMIT + 1)
    except HTTPError as err:
        message = _provider_message(err)
        if key:  # some providers quote the key they refused
            message = message.replace(key, "[key]")
        raise BackendError(
            f"HTTP {err.code} from {endpoint.base()}{path}{message}"
        ) from None
    except TimeoutError:
        raise BackendError(
            f"timed out after {timeout:g} s at {endpoint.base()}"
        ) from None
    except (URLError, OSError, HTTPException) as err:
        reason = getattr(err, "reason", None) or err
        raise BackendError(f"cannot reach {endpoint.base()}: {reason}") from None
    if len(raw) > RESPONSE_LIMIT:
        raise BackendError(
            f"response from {endpoint.base()} exceeds {RESPONSE_LIMIT >> 20} MiB"
        )
    try:
        data = json.loads(raw)
    except ValueError:
        raise BackendError(f"{endpoint.base()}{path} did not return JSON") from None
    if not isinstance(data, dict):
        raise BackendError(
            f"{endpoint.base()}{path} returned JSON that is not an object"
        )
    return data


def _provider_message(err: HTTPError) -> str:
    """The provider's own short error message (`error.message`), never the raw body."""
    try:
        data = json.loads(err.read(65536))
        message = (
            data.get("error", {}).get("message")
            if isinstance(data.get("error"), dict)
            else data.get("error")
        )
    except (ValueError, AttributeError, OSError):
        return ""
    if not isinstance(message, str) or not message.strip():
        return ""
    message = " ".join(message.split())
    return f": {message[:ERROR_CHARS]}{'…' if len(message) > ERROR_CHARS else ''}"


def _count(value: object) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _openai(r: Route, system: str, prompt: str, timeout: float) -> tuple[str, dict]:
    assert r.endpoint is not None
    payload: dict[str, Any] = {
        "model": r.model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    if r.endpoint.max_tokens is not None:
        payload["max_tokens"] = r.endpoint.max_tokens
    payload.update(r.endpoint.extra_body)
    data = request(r.endpoint, "/chat/completions", payload, timeout)
    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise BackendError(f"{r.model}: the reply has no choices[0].message") from None
    finish = choice.get("finish_reason")
    if finish not in (None, "stop", "end_turn"):
        raise BackendError(f"{r.model}: stopped: {finish}")
    if isinstance(content, list):  # some servers return content parts
        content = "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    if not isinstance(content, str):
        raise BackendError(f"{r.model}: the reply has no text")
    raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    cached = _count(
        (raw.get("prompt_tokens_details") or {}).get("cached_tokens")
    ) or _count(raw.get("prompt_cache_hit_tokens"))
    usage = zero_usage()
    usage["input"] = max(_count(raw.get("prompt_tokens")) - cached, 0)
    usage["cacheRead"] = cached
    usage["output"] = _count(raw.get("completion_tokens"))
    usage["totalTokens"] = (
        _count(raw.get("total_tokens")) or usage["input"] + cached + usage["output"]
    )
    return content, usage


def _anthropic(r: Route, system: str, prompt: str, timeout: float) -> tuple[str, dict]:
    assert r.endpoint is not None
    payload: dict[str, Any] = {
        "model": r.model_id,
        "max_tokens": r.endpoint.max_tokens or DEFAULT_MAX_TOKENS,
        # The prompt is the same on every call, so it is cached once it is long enough for the model's minimum.
        "system": [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(r.endpoint.extra_body)
    data = request(r.endpoint, "/v1/messages", payload, timeout)
    stop = data.get("stop_reason")
    if stop not in ("end_turn", "stop_sequence"):
        raise BackendError(f"{r.model}: stopped: {stop}")
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise BackendError(f"{r.model}: the reply has no content")
    text = "".join(
        b.get("text", "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )
    raw = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    usage = zero_usage()
    usage["input"] = _count(raw.get("input_tokens"))
    usage["output"] = _count(raw.get("output_tokens"))
    usage["cacheRead"] = _count(raw.get("cache_read_input_tokens"))
    usage["cacheWrite"] = _count(raw.get("cache_creation_input_tokens"))
    usage["totalTokens"] = (
        usage["input"] + usage["output"] + usage["cacheRead"] + usage["cacheWrite"]
    )
    return text, usage


def _pi(model: str, system: str, prompt: str, timeout: float) -> tuple[str, dict]:
    """One completion through `pi -p`. Every pi resource is off and `--system-prompt` replaces pi's own prompt, so
    the model sees memshort's prompt, one working-directory line, and the input; no tools are declared."""
    pi = shutil.which("pi")
    if pi is None:
        raise BackendError(
            "pi is not on PATH; install pi or name a model on a direct endpoint"
        )
    cmd = [
        pi, "-p",
        "--mode", "json",
        "--no-session", "--no-extensions", "--no-tools", "--no-mcp", "--no-skills",
        "--no-context-files", "--no-prompt-templates", "--no-themes", "--no-approve",
        "--model", model,
        "--thinking", "off",
        "--system-prompt", system,
        "Process the input above.",
    ]  # fmt: skip
    try:
        # Outside any project, so no project-local pi settings apply. The prompt goes on stdin: a transcript can be
        # longer than one argv string may be.
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise BackendError(f"pi timed out after {timeout:g} s") from None
    except OSError as err:
        raise BackendError(f"cannot run pi at {pi}: {err.strerror or err}") from None
    text, usage, error = _pi_reply(proc.stdout)
    if proc.returncode != 0 and not error:
        error = proc.stderr.strip()[-500:] or f"pi exited {proc.returncode}"
    if error:
        raise BackendError(error)
    return text, usage


def _pi_reply(out: str) -> tuple[str, dict, str]:
    message: dict = {}
    for line in out.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("type") == "message_end"
            and (event.get("message") or {}).get("role") == "assistant"
        ):
            message = event["message"]
    if not message:
        return "", {}, "no reply"
    if message.get("stopReason") not in ("stop", "end_turn"):
        return (
            "",
            {},
            message.get("errorMessage") or f"stopped: {message.get('stopReason')}",
        )
    text = "".join(
        b.get("text", "") for b in message.get("content", []) if b.get("type") == "text"
    )
    return text, message.get("usage") or {}, ""


def models(endpoint: Endpoint, timeout: float = 15) -> list[str]:
    """The model ids the endpoint lists: GET /models on `openai`, /v1/models on `anthropic`."""
    data = request(
        endpoint,
        "/v1/models" if endpoint.api == "anthropic" else "/models",
        None,
        timeout,
    )
    items = data.get("data")
    if not isinstance(items, list):
        raise BackendError(f"{endpoint.base()} returned no model list")
    return [
        m["id"] for m in items if isinstance(m, dict) and isinstance(m.get("id"), str)
    ]
