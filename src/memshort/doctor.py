"""Read-only readiness checks for the compaction models, for `memshort doctor`.

Each distinct model `memshort om` would use (`installation.stage_model`) is checked on its own route:

- `pi/...`: `pi --version` proves the executable runs, and `pi auth check --model ... --json` asks pi itself whether
  it can resolve credentials. Only `status`, `provider`, `authType` and `reason` are kept; subprocess stdout is never
  printed, `--credentials` is never passed, and `--no-refresh` is passed unless refresh is asked for.
- an HTTP endpoint: the key's environment variable is set when the endpoint needs one (its value is never shown),
  and the endpoint's model list is fetched to show it is reachable, the key is accepted, and the model is there.

`probe=True` sends one synthetic completion through each route, the way `memshort om` calls it. That is a model call
that costs a few tokens and may load a local model. No check sends session text.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass

from memshort import backends, installation

# A version check is local and should be instant; auth check may consult an OAuth provider, but not a model.
VERSION_TIMEOUT_S = 10
AUTH_TIMEOUT_S = 30


@dataclass(frozen=True)
class Check:
    """One readiness check. `detail` never contains a secret or a subprocess's raw stdout."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Report:
    """The resolved models and every check that was run. `ok` is true only if all checks passed.

    `models` maps each stage (or `model`, for an explicit one) to its model, None when it did not resolve."""

    models: dict[str, str | None]
    checks: tuple[Check, ...]

    @property
    def model(self) -> str | None:
        """The one model every stage uses, else None."""
        chosen = set(self.models.values())
        return chosen.pop() if len(chosen) == 1 else None

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def ready(self) -> bool:
        """The same answer as `ok`, named for the question a diagnostic command asks."""
        return self.ok

    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.ok)

    def render(self) -> str:
        """Human-readable lines; the CLI can print this, or format `checks` itself."""
        lines: list[str] = []
        lines.extend(
            f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}"
            for check in self.checks
        )
        lines.append("ready" if self.ok else "not ready")
        return "\n".join(lines)


def check(
    model: str | None = None, refresh: bool = False, probe: bool = False
) -> Report:
    """Check `model`, or every model `memshort om` would use, on its own route. Raises nothing: a missing pi, a bad
    selection or an unreachable endpoint is a failed check."""
    checks: list[Check] = []
    models, endpoints = _models(model, checks)
    distinct = list(dict.fromkeys(m for m in models.values() if m is not None))
    pi: tuple[str | None, Check] | None = (
        None  # the pi executable, checked once for every pi model
    )
    for chosen in distinct:
        suffix = f" ({chosen})" if len(distinct) > 1 else ""
        r = backends.route(chosen, endpoints)
        if r.endpoint is None:
            if pi is None:
                pi = _pi_executable()
                checks.append(pi[1])
            path = pi[0]
            if path is None:
                continue
            found = [_credentials(path, r.model_id, refresh)]
        else:
            found = _http_checks(r)
        if probe and all(c.ok for c in found):
            found.append(_probe(r))
        checks.extend(Check(c.name + suffix, c.ok, c.detail) for c in found)
    return Report(models, tuple(checks))


def _models(
    model: str | None, checks: list[Check]
) -> tuple[dict[str, str | None], dict[str, backends.Endpoint]]:
    """Resolve and validate the models; an explicit one is validated the same way without saving it."""
    try:
        selection = installation.load()
    except installation.SelectionError as err:
        for stage in installation.STAGES if model is None else ("model",):
            checks.append(
                Check(
                    f"{stage} model", False, f"cannot read the install selection: {err}"
                )
            )
        return (
            {stage: None for stage in installation.STAGES}
            if model is None
            else {"model": None}
        ), {}
    endpoints = selection.endpoint_table()
    if model is not None:
        try:
            chosen = backends.route(model, endpoints).model
        except backends.BackendError as err:
            checks.append(Check("compaction model", False, str(err)))
            return {"model": None}, endpoints
        checks.append(
            Check("compaction model", True, f"{chosen} via {_via(chosen, endpoints)}")
        )
        return {"model": chosen}, endpoints
    out: dict[str, str | None] = {}
    for stage in installation.STAGES:
        try:
            chosen, source = installation.stage_model(stage, selection)
        except installation.SelectionError as err:
            checks.append(
                Check(
                    f"{stage} model",
                    False,
                    f"cannot read the {stage} model from the install selection: {err}",
                )
            )
            out[stage] = None
            continue
        checks.append(
            Check(
                f"{stage} model",
                True,
                f"{chosen} (from {source}) via {_via(chosen, endpoints)}",
            )
        )
        out[stage] = chosen
    return out, endpoints


def _via(model: str, endpoints: dict[str, backends.Endpoint]) -> str:
    r = backends.route(model, endpoints)
    return "pi" if r.endpoint is None else r.endpoint.describe()


def _pi_executable() -> tuple[str | None, Check]:
    """pi's path when it runs, and the check that says so."""
    pi = shutil.which("pi")
    if pi is None:
        return None, Check(
            "pi executable",
            False,
            "pi is not on PATH; install pi or name a model on a direct endpoint",
        )
    version, error = _version(pi)
    if error:
        return None, Check("pi executable", False, error)
    return pi, Check("pi executable", True, f"{pi} ({version})")


def _http_checks(r: backends.Route) -> list[Check]:
    endpoint = r.endpoint
    assert endpoint is not None
    try:
        key = endpoint.key()
    except backends.BackendError as err:
        return [Check("API key", False, str(err))]
    checks = []
    if endpoint.api_key_env:
        used = (
            next(v for v in endpoint.api_key_env if os.environ.get(v, "").strip())
            if key
            else None
        )
        checks.append(
            Check(
                "API key",
                True,
                f"set in {used}" if used else "none set; sent without a key",
            )
        )
    try:
        listed = backends.models(endpoint)
    except backends.BackendError as err:
        if "HTTP 404" in str(err) or "HTTP 405" in str(err):
            checks.append(
                Check(
                    "endpoint",
                    True,
                    f"{endpoint.base()} has no model list; use --probe to test a call",
                )
            )
        else:
            checks.append(Check("endpoint", False, str(err)))
        return checks
    if r.model_id in listed:
        checks.append(Check("endpoint", True, f"{endpoint.base()} lists {r.model_id}"))
    else:
        sample = ", ".join(sorted(listed)[:8]) + (", ..." if len(listed) > 8 else "")
        checks.append(
            Check(
                "endpoint",
                False,
                f"{endpoint.base()} does not list {r.model_id}; it lists {sample or 'nothing'}",
            )
        )
    return checks


PROBE_SYSTEM = "You are a connectivity test. Follow the instruction exactly."
PROBE_PROMPT = "Reply with the single word READY."


def _probe(r: backends.Route) -> Check:
    try:
        text, usage = backends.complete(r, PROBE_SYSTEM, PROBE_PROMPT, 120)
    except backends.BackendError as err:
        return Check("probe", False, str(err))
    if not text.strip():
        return Check("probe", False, "the model replied with no text")
    tokens = (usage or {}).get("totalTokens") or 0
    return Check(
        "probe",
        True,
        f"replied with synthetic input ({tokens} tokens)" if tokens else "replied",
    )


def _version(pi: str) -> tuple[str, str]:
    """The version string pi prints, or the reason it could not be run."""
    try:
        ran = subprocess.run(
            [pi, "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "", f"pi --version timed out after {VERSION_TIMEOUT_S} s"
    except OSError as err:
        return "", f"cannot run pi at {pi}: {err.strerror or err}"
    if ran.returncode != 0:
        return "", f"pi --version exited {ran.returncode}"
    version = next(
        (line.strip() for line in (ran.stdout or "").splitlines() if line.strip()), ""
    )
    return version or "version unknown", ""


def _credentials(pi: str, model: str, refresh: bool) -> Check:
    """Ask pi's own documented credential check; keep only its structured, secret-free fields."""
    cmd = [pi, "auth", "check", "--model", model, "--json"]
    if not refresh:
        cmd.append("--no-refresh")
    try:
        ran = subprocess.run(
            cmd, capture_output=True, text=True, timeout=AUTH_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired:
        return Check(
            "model credentials",
            False,
            f"pi auth check timed out after {AUTH_TIMEOUT_S} s",
        )
    except OSError as err:
        return Check(
            "model credentials", False, f"cannot run pi at {pi}: {err.strerror or err}"
        )
    data = _last_json_object(ran.stdout)
    if data is None:
        # Raw stdout is deliberately not included: this path must not become a way to print a credential.
        return Check(
            "model credentials",
            False,
            f"pi auth check returned no JSON (exit {ran.returncode})",
        )
    status = data.get("status")
    provider = data.get("provider")
    provider = provider if isinstance(provider, str) and provider else model
    auth_type = data.get("authType")
    suffix = f" ({auth_type})" if isinstance(auth_type, str) and auth_type else ""
    if ran.returncode == 0 and status == "ready":
        return Check("model credentials", True, f"ready for {provider}{suffix}")
    if status in ("not_ready", "invalid"):
        reason = data.get("reason")
        why = f": {reason}" if isinstance(reason, str) and reason else ""
        return Check(
            "model credentials", False, f"{status} for {provider}{suffix}{why}"
        )
    return Check(
        "model credentials",
        False,
        f"unexpected pi auth status {status!r} (exit {ran.returncode})",
    )


def _last_json_object(out: str) -> dict | None:
    """The last JSON object in pi's output, ignoring any other line; never returns raw text to the caller."""
    for line in reversed((out or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
