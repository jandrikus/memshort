"""memshort in Codex: two lifecycle hooks around Codex's own compaction.

Codex has no hook that can replace its compaction summary (rust-v0.167.1: `PreCompact` and `PostCompact` take no
history back). What it has is enough to add memshort's log next to that summary:

- `PreCompact` runs before Codex compacts. `memshort codex-hook pre-compact` reads the rollout JSONL the hook names
  (`transcript_path`), turns every line it has not observed yet into transcript text and runs the observer (and the
  reflector when the log is over its limit) in this process. The log, the task and how far the rollout was observed
  are saved per session under `$XDG_STATE_HOME/memshort/codex/`.
- `Stop` runs after each turn, in the background (`async`). Once `observe_every_tokens` of the rollout is
  unobserved it observes the same way, so a compaction usually finds most of the session observed already. A lock
  per session keeps the two from observing at once.
- `SessionStart` with source `compact` runs after the compaction, before the next model request, also in the middle
  of a turn. `memshort codex-hook session-start` prints the saved log as `additionalContext`, which Codex adds as a
  developer message. Codex drops that message at its next compaction (its replacement history keeps only the user
  messages and its summary), so each compaction gets the log once, current as of that compaction.

A failed observation does not stop Codex: its own summary stands in, the `systemMessage` says why, and the lines
stay unobserved, so the next compaction observes them. The rollout format is not a stable Codex interface; an item
type this module does not know is rendered as its JSON, not dropped. Hooks of subagents (they carry `agent_id` and
the parent's session id) are ignored.

The hooks run only while the install selection names `codex`.
"""

import fcntl
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from memshort import installation, om

ENV_HOME = "CODEX_HOME"
HOOK_COMMAND = "codex-hook"
EVENTS = {
    # event: (matcher, our sub-command, timeout in seconds, extra handler keys)
    # PreCompact observes synchronously, so its result is there when SessionStart reads it. A first compaction of a
    # long session can take several om runs; each one is saved as it finishes, so a timeout loses only the last.
    "PreCompact": ("auto|manual", "pre-compact", 1800, {}),
    # memshort bounds the log itself (`condense_log_above_tokens`), so Codex passes it whole instead of spilling it.
    "SessionStart": ("^compact$", "session-start", 30, {"additionalContextLimit": 0}),
    # After each turn, in the background: observe once `observe_every_tokens` of the rollout is unobserved.
    "Stop": ("", "stop", 1800, {"async": True}),
}
# One om run gets at most this much of the unobserved rollout, as in the Hermes engine.
MAX_OBSERVE_TOKENS = 30_000
TOOL_RESULT_CAP = 2_000
MESSAGE_CAP = 20_000
# Context Codex itself puts into user messages; it is not conversation.
CODEX_CONTEXT = (
    "<environment_context>",
    "<user_instructions>",
    "# AGENTS.md instructions",
)
STATE_VERSION = 1


def codex_home() -> Path:
    override = os.environ.get(ENV_HOME)
    return Path(override).expanduser() if override else Path.home() / ".codex"


def hooks_path() -> Path:
    return codex_home() / "hooks.json"


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".local/state"
    return root / "memshort" / "codex"


# ── the rollout as transcript text ────────────────────────────────────────────────────────────────


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content, ensure_ascii=False)
    out: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            out.append(part["text"])
        elif str(part.get("type", "")).endswith("image"):
            out.append("(image)")
    return "\n".join(out)


def _clip(text: str, cap: int) -> str:
    return (
        f"{text[:cap]}… ({len(text) - cap} characters cut)" if len(text) > cap else text
    )


def render_item(item: dict) -> str | None:
    """One rollout `response_item` as an observer line; None for what is not conversation."""
    kind = item.get("type")
    if kind == "message":
        role = item.get("role")
        text = _text(item.get("content")).strip()
        if role == "user":
            if not text or text.startswith(CODEX_CONTEXT):
                return None
            return f"[User]: {_clip(text, MESSAGE_CAP)}"
        if role == "assistant":
            return f"[Assistant]: {_clip(text, MESSAGE_CAP)}" if text else None
        return None  # developer and system messages: Codex's instructions and hook context, ours included
    if kind in ("reasoning", "compaction", "ghost_snapshot"):
        return None
    if kind in ("function_call", "custom_tool_call"):
        arguments = item.get("arguments", item.get("input", ""))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        return (
            f"[Tool call {item.get('name') or 'tool'}: {_clip(arguments, MESSAGE_CAP)}]"
        )
    if kind in ("function_call_output", "custom_tool_call_output"):
        output = item.get("output")
        if isinstance(output, dict):
            output = output.get("content", output)
        return f"[Tool result]: {_clip(_text(output), TOOL_RESULT_CAP)}"
    rest = {
        k: v
        for k, v in item.items()
        if k not in ("id", "internal_chat_message_metadata_passthrough")
    }
    return f"[{kind}]: {_clip(json.dumps(rest, ensure_ascii=False), TOOL_RESULT_CAP)}"


def read_rollout(path: Path, start: int) -> tuple[list[str], int]:
    """Observer lines for the rollout from line `start` on, and the number of complete lines read.

    Codex's local compaction records its summary as an assistant message right before the `compacted` record; that
    is Codex's summary, not conversation, so it is left out. A last line without its newline is still being written
    and is read next time.
    """
    with path.open("rb") as stream:
        data = stream.read()
    raw = data.split(b"\n")
    complete = raw[
        :-1
    ]  # the part after the last newline is empty or still being written
    lines: list[str] = []
    last_was_assistant = False
    for record_bytes in complete[start:]:
        try:
            record = json.loads(record_bytes)
        except ValueError:
            raise om.ModelError(
                f"{path} is not a JSONL rollout this memshort can read"
            ) from None
        kind = record.get("type") if isinstance(record, dict) else None
        payload = record.get("payload") if isinstance(record, dict) else None
        if kind == "compacted":
            if last_was_assistant and lines:
                lines.pop()
            last_was_assistant = False
            continue
        if kind != "response_item" or not isinstance(payload, dict):
            continue
        line = render_item(payload)
        if line is None:
            continue
        lines.append(line)
        last_was_assistant = line.startswith("[Assistant]:")
    return lines, len(complete)


def chunks(lines: list[str], max_tokens: int | None = None) -> list[str]:
    max_tokens = max_tokens or MAX_OBSERVE_TOKENS
    out: list[str] = []
    current: list[str] = []
    used = 0
    for line in lines:
        size = om.tokens(line)
        if current and used + size > max_tokens:
            out.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += size
    if current:
        out.append("\n".join(current))
    return out


# ── per-session state ─────────────────────────────────────────────────────────────────────────────


def _state_path(session_id: str, transcript: str) -> Path:
    key = hashlib.sha256(f"{session_id}\0{transcript}".encode()).hexdigest()[:24]
    return state_dir() / f"{key}.json"


def _read_state(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as err:
        raise om.ModelError(f"cannot read {path}: {err}") from err
    if not isinstance(data, dict) or data.get("version") != STATE_VERSION:
        raise om.ModelError(f"{path} is not a memshort Codex state file")
    return data


def _write_state(path: Path, state: dict) -> None:
    # Session text: the directory and the file are private from the moment they exist.
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"version": STATE_VERSION, **state}, stream, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """One observation per session at a time: a second PreCompact waits for the first."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# ── the hooks ─────────────────────────────────────────────────────────────────────────────────────


def _selected() -> bool:
    """Only an explicit selection that names codex turns the hooks on; a broken one is reported, not guessed."""
    if not installation.exists():
        return False
    return "codex" in installation.load().harnesses


def observe(event: dict, at_least: int = 0, now: datetime | None = None) -> dict:
    """Observe what the rollout holds beyond the saved state, when that is at least `at_least` tokens."""
    transcript = event.get("transcript_path")
    session_id = event.get("session_id")
    if event.get("agent_id") or not isinstance(session_id, str) or not session_id:
        return {}
    if not isinstance(transcript, str) or not transcript:
        return {
            "systemMessage": "memshort: Codex gave no transcript path, nothing observed"
        }
    path = _state_path(session_id, transcript)
    with _locked(path):
        state = _read_state(path)
        mark = int(state.get("through_line", 0))
        lines, read = read_rollout(Path(transcript), mark)
        if at_least and om.tokens("\n".join(lines)) < at_least:
            return {}
        log, task = state.get("log", ""), state.get("task", "")
        pieces = chunks(lines) or [""]
        for index, piece in enumerate(pieces):
            if piece:
                result = om.run({"log": log, "task": task, "transcript": piece}, now)
                log, task = result["log"], result["task"]
            # Only the last piece moves the mark: a run cut short re-observes the rest instead of skipping it.
            done = read if index == len(pieces) - 1 else mark
            _write_state(
                path,
                {
                    "log": log,
                    "task": task,
                    "through_line": done,
                    "updated_at": time.time(),
                },
            )
    return {}


def pre_compact(event: dict, now: datetime | None = None) -> dict:
    return observe(event, 0, now)


def stop(event: dict, now: datetime | None = None) -> dict:
    """Observe in the background after a turn, once enough is unobserved, so a compaction has less left to do."""
    return observe(event, installation.threshold("observe_every_tokens")[0], now)


INTRO = (
    "memshort observed this session before Codex compacted it. This is its log of the conversation up to that "
    "compaction, oldest first. It is more detailed than the summary above; where they disagree, trust the log."
)


def session_start(event: dict) -> dict:
    session_id, transcript = event.get("session_id"), event.get("transcript_path")
    if (
        event.get("source") != "compact"
        or not isinstance(session_id, str)
        or not isinstance(transcript, str)
    ):
        return {}
    state = _read_state(_state_path(session_id, transcript))
    log, task = state.get("log", ""), state.get("task", "")
    if not log.strip():
        return {}
    text = (
        "<memshort-observations>\n"
        f"{INTRO}\n"
        "🔴 the user's requests, decisions and constraints; 🟡 work done and findings; 🟢 minor context.\n\n"
        f"<observation-log>\n{log.strip()}\n</observation-log>\n\n"
        f"<current-task>\n{task.strip() or '(none)'}\n</current-task>\n"
        "</memshort-observations>"
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": text,
        }
    }


HANDLERS = {"pre-compact": pre_compact, "session-start": session_start, "stop": stop}


def hook(name: str, stdin: str) -> tuple[dict, int]:
    """Run one hook: its JSON output and exit code. A failure is a warning; Codex's own compaction stands."""
    try:
        event = json.loads(stdin or "{}")
        if not isinstance(event, dict):
            raise TypeError("hook input is not a JSON object")
        if not _selected():
            return {}, 0
        return HANDLERS[name](event), 0
    except (
        om.ModelError,
        installation.SelectionError,
        OSError,
        TypeError,
        ValueError,
    ) as err:
        stage = "log injection" if name == "session-start" else "observation"
        return {
            "systemMessage": f"memshort: {stage} failed, Codex's own summary stands in: {err}"
        }, 0


def main(name: str) -> None:
    output, code = hook(name, sys.stdin.read())
    if output:
        print(json.dumps(output, ensure_ascii=False))
    raise SystemExit(code)


# ── install ───────────────────────────────────────────────────────────────────────────────────────


def command(sub: str, binary: str | None = None) -> str:
    binary = binary or shutil.which("memshort") or "memshort"
    return f"{shlex.quote(binary)} {HOOK_COMMAND} {sub}"


def ours(handler: Any) -> bool:
    return (
        isinstance(handler, dict)
        and isinstance(handler.get("command"), str)
        and f" {HOOK_COMMAND} " in handler["command"]
    )


def read_hooks(path: Path) -> dict:
    if path.is_symlink():
        raise ValueError(f"{path} is a symlink; left in place")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("hooks", {}), dict):
        raise TypeError(f"{path} must be an object with a hooks object")
    return data


def without_ours(data: dict) -> dict:
    """The hooks file with every memshort handler taken out, and groups or events left empty by that removed."""
    out = dict(data)
    hooks: dict = {}
    for event, groups in data.get("hooks", {}).items():
        if not isinstance(groups, list):
            hooks[event] = groups
            continue
        kept = []
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("hooks"), list):
                handlers = [h for h in group["hooks"] if not ours(h)]
                if not handlers:
                    continue
                group = {**group, "hooks": handlers}
            kept.append(group)
        if kept:
            hooks[event] = kept
    if hooks or "hooks" in data:
        out["hooks"] = hooks
    return out


def with_ours(data: dict, binary: str | None = None) -> dict:
    out = without_ours(data)
    hooks = dict(out.get("hooks", {}))
    for event, (matcher, sub, timeout, extra) in EVENTS.items():
        handler = {
            "type": "command",
            "command": command(sub, binary),
            "timeout": timeout,
            **extra,
        }
        group = (
            {"matcher": matcher, "hooks": [handler]}
            if matcher
            else {"hooks": [handler]}
        )
        hooks[event] = [*hooks.get(event, []), group]
    out["hooks"] = hooks
    return out


def write_hooks(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o600
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
