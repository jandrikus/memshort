"""Hermes context engine: observational compaction for live sessions.

Older messages are replaced by a dated observation log that `memshort om` writes; the newest ~15k tokens stay raw.
A completed user turn over 100k raw tokens is compacted at the next preflight. During a turn, compaction starts
at 200k raw tokens or the model's safety threshold. Every 10k tokens older than the kept tail are observed in the
background and persisted under Hermes' plugin-data directory, so compaction usually only observes the delta.
Those are the defaults of `THRESHOLDS`; the host's memshort selection and each threshold's environment variable
override them, as in memshort's installation.py.

The `<memshort-observations>` summary in the transcript is authoritative after a compaction. It is a user turn (the
first visible message after the system/developer head, as Anthropic and Bedrock require). If the kept tail starts
with a user turn, a fixed assistant acknowledgement separates it from the summary.
Persisted background state is used only when its session, generation, base-summary digest and covered-prefix
digest all match the messages handed in; anything else is treated as absent. A worker that finishes after a reset
or a session switch is discarded and its `memshort om` subprocess is killed instead of spending more tokens.

This module imports only the standard library. `memshort om` runs as a subprocess, so no memshort code enters Hermes'
environment and the plugin has no compiled dependencies. Token sizes use Hermes' rough estimate (bytes/CJK, flat
image cost), not an exact tokenizer; digit- and JSON-heavy text can still be underestimated. A compaction failure
raises with memshort's reason and leaves the original messages untouched: there is no fallback summary.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

PLUGIN_NAME = "memshort"
# The same table as THRESHOLDS in memshort's installation.py: file key, environment variable, default tokens.
# The engine uses the first four; `memshort om` reads the reflector's limit itself.
THRESHOLDS: dict[str, tuple[str, int]] = {
    "compact_after_turn_tokens": ("MEMSHORT_COMPACT_AFTER_TURN_TOKENS", 100_000),
    "compact_during_turn_tokens": ("MEMSHORT_COMPACT_DURING_TURN_TOKENS", 200_000),
    "keep_recent_tokens": ("MEMSHORT_KEEP_RECENT_TOKENS", 15_000),
    "observe_every_tokens": ("MEMSHORT_OBSERVE_EVERY_TOKENS", 10_000),
    "condense_log_above_tokens": ("MEMSHORT_CONDENSE_LOG_ABOVE_TOKENS", 60_000),
}
# Raw tokens after the last compaction that start the next one.
DEFAULT_ACTIVATE_TOKENS = THRESHOLDS["compact_after_turn_tokens"][1]
DEFAULT_MIDTURN_TOKENS = THRESHOLDS["compact_during_turn_tokens"][1]
# The newest raw tokens a compaction keeps as they are.
DEFAULT_KEEP_TOKENS = THRESHOLDS["keep_recent_tokens"][1]
# Unobserved tokens older than the kept tail that start a background observation.
DEFAULT_BUFFER_TOKENS = THRESHOLDS["observe_every_tokens"][1]
# A foreground `memshort om` must finish inside Hermes' compression inactivity budget (120 s by default), or the
# host abandons the pass while the CLI keeps running. 100 s leaves room for the failure to surface.
DEFAULT_FOREGROUND_TIMEOUT_S = 100.0
# A background observation is not waited on by the host; 1200 s covers `memshort om`'s own retry bounds
# (two completions of up to 300 s each, observer plus reflector).
DEFAULT_BACKGROUND_TIMEOUT_S = 1_200.0
# One om call gets at most this much of the unobserved slice; a cold long session is observed in order.
MAX_OBSERVE_TOKENS = 30_000
CHARS_PER_TOKEN = 4
# Hermes' flat per-image estimate before the host's usage calibration (agent/image_token_cost.py).
IMAGE_TOKENS = 1_500
_CJK_DENSE_RE = re.compile(
    "[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]"
)
FAILURE_COOLDOWN_S = 120.0
# Truncation of the observer's input. Tool output is where the volume is; pi's serializer cuts it the same way.
TOOL_RESULT_CAP = 2_000
MESSAGE_CAP = 20_000

SUMMARY_OPEN = "<memshort-observations>"
SUMMARY_CLOSE = "</memshort-observations>"
# A log written before the rename is still this engine's own summary.
LEGACY_OPEN = "<aimem-observations>"
LEGACY_CLOSE = "</aimem-observations>"
SUMMARY_TAG = SUMMARY_OPEN
# The summary is a standalone user turn (Anthropic/Bedrock require the first visible role to be "user"). A fixed
# assistant acknowledgement follows it so the summary and the first kept user turn do not become adjacent users;
# it is skipped again when the transcript is read, so it never enters the next observation log.
ACK_CONTENT = "Understood — continuing from these observations."
LOG_TAG = "observation-log"
# The verbatim blocks `memshort om` keeps inside the log (om.py, `split_log`).
BLOCK_OPEN = "<memshort-preserved>"
BLOCK_CLOSE = "</memshort-preserved>"
TASK_TAG = "current-task"
LOG_RE = re.compile(rf"<{LOG_TAG}>\s*(.*?)\s*</{LOG_TAG}>", re.DOTALL)
TASK_RE = re.compile(rf"<{TASK_TAG}>\s*(.*?)\s*</{TASK_TAG}>", re.DOTALL)
VISIBLE_ROLES = ("user", "assistant")
HEAD_ROLES = ("system", "developer")
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()

Runner = Callable[[dict[str, Any], float], dict[str, Any]]


class HermesCompactionError(RuntimeError):
    """A visible compaction failure. The host keeps the original transcript; no summary is substituted."""


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("ignoring %s=%r: not a number", name, raw)
        return default
    return value if value > 0 else default


def _token_count(value: Any) -> int | None:
    """A positive whole number of tokens, as memshort's installation.py accepts it; None when it is not."""
    if isinstance(value, str):
        text = value.strip().replace("_", "")
        value = int(text) if text.isdigit() else value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _selection_path() -> Path:
    override = os.environ.get("MEMSHORT_COMPACT_INSTALL_CONFIG")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "memshort" / "install.json"


def thresholds() -> dict[str, int]:
    """Every threshold in effect: its environment variable, else the host's selection, else the default.

    A bad value raises `HermesCompactionError` naming the variable or the file; it never falls back silently.
    The rest of the selection is the installer's to check, so only `thresholds` is read here.
    """
    path = _selection_path()
    own: dict[str, Any] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as err:
        raise HermesCompactionError(f"cannot read {path}: {err}") from err
    if isinstance(data, dict):
        own = data.get("thresholds", {})
    if not isinstance(own, dict):
        raise HermesCompactionError(f"{path}: thresholds must be an object")
    unknown = sorted(set(own) - set(THRESHOLDS))
    if unknown:
        raise HermesCompactionError(
            f"{path}: unknown thresholds: {', '.join(unknown)}; known: {', '.join(THRESHOLDS)}"
        )
    out: dict[str, int] = {}
    for key, (variable, default) in THRESHOLDS.items():
        raw = os.environ.get(variable, "").strip()
        source, value = (
            (variable, raw) if raw else (f"{path}: {key}", own.get(key, default))
        )
        n = _token_count(value)
        if n is None:
            raise HermesCompactionError(
                f"{source} must be a positive whole number of tokens, not {value!r}"
            )
        out[key] = n
    return out


def _positive_int(value: Any, default: int) -> int:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else default
    )


def _positive_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return default


def tokens_of(text: str) -> int:
    """Hermes' rough token estimate, mirrored locally: CJK ~1 token/char, otherwise ceil(UTF-8 bytes / 4).

    This is not a tokenizer: command lines, paths and JSON tokenize denser than bytes/4, so the raw
    trigger and the kept-tail budget can underestimate real provider tokens. Hermes exposes no exact counter to
    plugins; using this instead of chars/4 mostly fixes non-ASCII text, and images are priced separately below.
    """
    if not text:
        return 0
    text = str(text)
    if text.isascii():
        return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN
    stripped = _CJK_DENSE_RE.sub("", text)
    dense = len(text) - len(stripped)
    return dense + (
        (len(stripped.encode("utf-8", "replace")) + CHARS_PER_TOKEN - 1)
        // CHARS_PER_TOKEN
    )


def _image_count(content: Any) -> int:
    if not isinstance(content, list):
        return 0
    return sum(
        1
        for block in content
        if isinstance(block, dict)
        and block.get("type") in ("image", "image_url", "input_image")
    )


def content_text(content: Any) -> str:
    """Flatten an OpenAI content field to text, keeping multimodal placeholders visible."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind in ("text", "input_text", "output_text") and isinstance(
                    block.get("text"), str
                ):
                    parts.append(block["text"])
                elif kind in ("image", "image_url", "input_image"):
                    parts.append("[image]")
                else:
                    parts.append(json.dumps(block, ensure_ascii=True, default=str))
            else:
                parts.append(str(block))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=True, default=str)
    return str(content)


def _plain_text(content: Any) -> str:
    """Only the text blocks of a content field, without placeholders for images or other blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content_text(content)
    return "\n".join(
        block if isinstance(block, str) else block["text"]
        for block in content
        if isinstance(block, str)
        or (
            isinstance(block, dict)
            and block.get("type") in ("text", "input_text", "output_text")
            and isinstance(block.get("text"), str)
        )
    )


def tool_calls_text(tool_calls: Any) -> str:
    lines: list[str] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = (
            call.get("function") if isinstance(call.get("function"), dict) else {}
        )
        name = function.get("name") or call.get("name") or "tool"
        arguments = function.get("arguments", call.get("arguments", ""))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=True, default=str)
        lines.append(f"[Tool call {name}: {arguments[:MESSAGE_CAP]}]")
    return "\n".join(lines)


def is_summary(message: Any) -> bool:
    """Whether a message is a complete memshort compaction summary, judged by its content only."""
    if not isinstance(message, dict) or message.get("role") not in VISIBLE_ROLES:
        return False
    text = content_text(message.get("content")).lstrip()
    opened = text.startswith((SUMMARY_OPEN, LEGACY_OPEN))
    closed = SUMMARY_CLOSE in text or LEGACY_CLOSE in text
    return opened and closed and f"<{LOG_TAG}>" in text


def log_span(text: str) -> tuple[int, int] | None:
    """Where the log's text starts and ends in a rendered summary, as `memshort om` reads it: line by line, each
    preserved block stepped over by its length, so a closing tag quoted inside one does not end the log."""
    start = text.find(f"<{LOG_TAG}>\n")
    if start < 0:
        return None
    start += len(LOG_TAG) + 3
    pos = start
    while pos < len(text):
        if text.startswith(BLOCK_OPEN, pos):
            try:
                end = text.index("\n", pos)
                length = json.loads(text[pos + len(BLOCK_OPEN) : end])["length"]
            except (ValueError, KeyError, TypeError):
                return None
            if not isinstance(length, int) or isinstance(length, bool) or length < 0:
                return None
            pos = end + 1 + length + 1
            if not text.startswith(BLOCK_CLOSE, pos):
                return None
            pos += len(BLOCK_CLOSE)
        else:
            line_end = text.find("\n", pos)
            line_end = len(text) if line_end < 0 else line_end
            if text[pos:line_end] == f"</{LOG_TAG}>":
                return start, max(start, pos - 1)
            pos = line_end
        if not text.startswith("\n", pos):
            return None
        pos += 1
    return None


def parse_summary(text: str) -> tuple[str, str]:
    """The log and current task of a rendered summary; empty strings when the text is not one."""
    if not (SUMMARY_OPEN in text or LEGACY_OPEN in text):
        return "", ""
    if SUMMARY_CLOSE not in text and LEGACY_CLOSE not in text:
        return "", ""
    span = log_span(text)
    if span is None:
        found = LOG_RE.search(text)
        log = found.group(1).strip() if found else ""
        task_from = found.end() if found else 0
    else:
        log = text[span[0] : span[1]].strip("\n")
        task_from = span[1]
    task_match = TASK_RE.search(text, task_from)
    task = task_match.group(1).strip() if task_match else ""
    return ("" if log.strip() == "(empty)" else log), ("" if task == "(none)" else task)


def is_ack(message: Any) -> bool:
    """Whether a message is the fixed assistant acknowledgement that follows a summary."""
    return (
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and content_text(message.get("content")).strip() == ACK_CONTENT
    )


SKILL_SCAFFOLD = ("[IMPORTANT: The user has invoked the ", '[IMPORTANT: The "')
_SKILL_NAME_RE = re.compile(r'^\[IMPORTANT: The (?:user has invoked the )?"([^"]*)"')


def _skill_view_reads_skill(call: dict[str, Any]) -> bool:
    """A `skill_view` call that returns the SKILL.md itself, not one of its linked files."""
    function = call.get("function") or {}
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return False
    path = args.get("file_path") if isinstance(args, dict) else None
    return not path or (isinstance(path, str) and Path(path).name == "SKILL.md")


def transcript_parts(
    messages: list[dict[str, Any]], first: bool
) -> list[dict[str, Any]]:
    """The slice as ordered transcript parts for `memshort om`: conversation runs, and between them the text kept
    verbatim in the log. That is the first user message when the slice starts the session (`first`), a user turn
    carrying Hermes' slash-skill, bundle or auto-load scaffold, and the content of a successful `skill_view` of a
    SKILL.md. A kept text is not also sent as conversation; an image in the first user message is noted."""
    parts: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []
    calls: set[str] = set()

    def flush() -> None:
        text = serialize_messages(run)
        run.clear()
        if text.strip():
            parts.append({"text": text})

    def keep(kind: str, content: str, name: str | None = None) -> None:
        flush()
        if content.strip():
            item = {"kind": kind, "content": content}
            if name:
                item["name"] = name
            parts.append({"preserved": item})

    for message in messages:
        if not isinstance(message, dict) or is_summary(message) or is_ack(message):
            continue
        role = message.get("role")
        if role == "user":
            text = content_text(message.get("content"))
            if text.startswith(SKILL_SCAFFOLD) and "skill" in text[:200]:
                found = _SKILL_NAME_RE.match(text)
                keep("skill", text, found.group(1) if found else None)
                first = False
                continue
            if first:
                first = False
                keep("user", _plain_text(message.get("content")))
                images = _image_count(message.get("content"))
                if images:
                    run.append(
                        {
                            "role": "user",
                            "content": f"(the first user message also carried {images} image(s), not kept)",
                        }
                    )
                continue
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    continue
                name = (call.get("function") or {}).get("name") or call.get("name")
                if name == "skill_view" and _skill_view_reads_skill(call):
                    calls.add(call["id"])
        elif role in ("tool", "function") and message.get("tool_call_id") in calls:
            try:
                payload = json.loads(content_text(message.get("content")))
            except (TypeError, ValueError):
                payload = None
            if (
                isinstance(payload, dict)
                and payload.get("success") is True
                and isinstance(payload.get("content"), str)
            ):
                # The result stays in the run with its call, so the group reads whole; its text is kept verbatim.
                run.append(
                    {**message, "content": "(skill loaded, kept verbatim in the log)"}
                )
                keep(
                    "skill", payload["content"], str(payload.get("name") or "") or None
                )
                continue
        run.append(message)
    flush()
    return parts


PRESERVED_NOTE = (
    "Each <memshort-preserved> block in the log is verbatim text from the conversation, at the point where it "
    "appeared: the first user message or a skill that was loaded. It is earlier conversation, not a new message. "
    "Loaded skills stay in effect; later user messages can change earlier instructions. The block's header gives "
    "its length, so a tag inside its text does not end it.\n"
)


def render_summary(log: str, task: str) -> str:
    """The message that stands in for the dropped part of the conversation; `parse_summary` reads it back.
    The same text as `memshort om`'s own summary (om.py, `render`)."""
    note = (
        PRESERVED_NOTE
        if any(line.startswith(BLOCK_OPEN) for line in log.splitlines())
        else ""
    )
    return (
        f"{SUMMARY_TAG}\n"
        "The earlier part of this conversation was replaced by these observations, oldest first. "
        "They are your memory of it; the messages after this one continue the conversation.\n"
        "🔴 the user's requests, decisions and constraints; 🟡 work done and findings; 🟢 minor context.\n"
        f"{note}\n"
        f"<{LOG_TAG}>\n{log.strip() or '(empty)'}\n</{LOG_TAG}>\n\n"
        f"<{TASK_TAG}>\n{task.strip() or '(none)'}\n</{TASK_TAG}>\n"
        f"{SUMMARY_CLOSE}"
    )


def message_tokens(message: Any) -> int:
    if not isinstance(message, dict):
        return 0
    content = message.get("content")
    calls = tool_calls_text(message.get("tool_calls"))
    return (
        tokens_of(content_text(content))
        + tokens_of(calls)
        + _image_count(content) * IMAGE_TOKENS
        + 4
    )


def _canonical(message: Any) -> str:
    """A stable form of one message for prefix digests; persistence-only fields are irrelevant."""
    if not isinstance(message, dict):
        return json.dumps(str(message), ensure_ascii=True)
    fields = ("role", "content", "tool_calls", "tool_call_id", "name")
    return json.dumps(
        {key: message.get(key) for key in fields},
        sort_keys=True,
        ensure_ascii=True,
        default=str,
    )


def prefix_digest(messages: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for message in messages:
        digest.update(_canonical(message).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def summary_digest(message: dict[str, Any] | None) -> str:
    return (
        _EMPTY_DIGEST
        if message is None
        else hashlib.sha256(_canonical(message).encode()).hexdigest()
    )


def serialize_messages(messages: list[dict[str, Any]]) -> str:
    """The observer's transcript: speaker-tagged, no quotes, our own summaries left out."""
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or is_summary(message) or is_ack(message):
            continue
        role = message.get("role")
        text = content_text(message.get("content"))
        cap = TOOL_RESULT_CAP if role in ("tool", "function") else MESSAGE_CAP
        if len(text) > cap:
            text = f"{text[:cap]}… ({len(text) - cap} characters cut)"
        if role == "user":
            lines.append(f"[User]: {text}" if text else "[User]:")
        elif role == "assistant":
            block = f"[Assistant]: {text}" if text else "[Assistant]:"
            calls = tool_calls_text(message.get("tool_calls"))
            lines.append(f"{block}\n{calls}" if calls else block)
        elif role in ("tool", "function"):
            name = message.get("name") or "tool"
            lines.append(
                f"[Tool result {name}]: {text}" if text else f"[Tool result {name}]:"
            )
        elif role in ("system", "developer"):
            lines.append(f"[System]: {text}" if text else "[System]:")
        else:
            lines.append(f"[{role}]: {text}" if text else f"[{role}]:")
    return "\n".join(lines)


@dataclass(frozen=True)
class Observation:
    """A completed background observation: the log and the exact messages it covers."""

    session_id: str
    instance_id: str
    generation: int
    base_digest: str
    covered_count: int
    through_digest: str
    log: str
    task: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "generation": self.generation,
            "base_digest": self.base_digest,
            "covered_count": self.covered_count,
            "through_digest": self.through_digest,
            "log": self.log,
            "task": self.task,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Observation | None:
        if not isinstance(raw, dict) or raw.get("version") != 1:
            return None
        try:
            observation = cls(
                session_id=str(raw["session_id"]),
                instance_id=str(raw["instance_id"]),
                generation=int(raw["generation"]),
                base_digest=str(raw["base_digest"]),
                covered_count=int(raw["covered_count"]),
                through_digest=str(raw["through_digest"]),
                log=str(raw["log"]),
                task=str(raw["task"]),
                updated_at=str(raw.get("updated_at") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if observation.covered_count < 0:
            return None
        return observation


@dataclass
class View:
    """Where a compaction would cut, and what the transcript's own summary already covers."""

    head_end: int
    start: int
    cut: int
    log: str
    task: str
    base_digest: str
    raw_tokens: int
    total_tokens: int

    @property
    def compactable(self) -> bool:
        return self.cut > self.start


def _summary_message(
    messages: list[dict[str, Any]], head_end: int
) -> tuple[int, dict[str, Any]] | None:
    found: tuple[int, dict[str, Any]] | None = None
    for index in range(head_end, len(messages)):
        if is_summary(messages[index]):
            found = (index, messages[index])
    return found


def _safe_cut(messages: list[dict[str, Any]], index: int) -> bool:
    """A kept tail may start at a user or assistant message, never inside a tool-call group."""
    message = messages[index]
    return (
        isinstance(message, dict)
        and message.get("role") in VISIBLE_ROLES
        and not is_summary(message)
    )


# If the safe cut older than the keep boundary would carry this many times the keep budget, cut at the next
# newer safe message instead. A kept tail smaller than the budget is allowed; the budget is a target, not a
# guarantee, because a tool-call group is never split.
MAX_TAIL_FACTOR = 2.0


def _cut_index(messages: list[dict[str, Any]], start: int, keep_tokens: int) -> int:
    kept = 0
    boundary: int | None = None
    for index in range(len(messages) - 1, start - 1, -1):
        kept += message_tokens(messages[index])
        if kept >= keep_tokens:
            boundary = index
            break
    if boundary is None:
        return start
    older = next(
        (i for i in range(boundary, start - 1, -1) if _safe_cut(messages, i)), None
    )
    newer = next(
        (i for i in range(boundary + 1, len(messages)) if _safe_cut(messages, i)), None
    )
    if older is None:
        return newer if newer is not None else start
    if newer is None or older == boundary:
        return older
    older_tokens = sum(message_tokens(message) for message in messages[older:])
    return older if older_tokens <= keep_tokens * MAX_TAIL_FACTOR else newer


def analyze(messages: list[dict[str, Any]], keep_tokens: int) -> View:
    head_end = 0
    while head_end < len(messages) and isinstance(messages[head_end], dict):
        if messages[head_end].get("role") not in HEAD_ROLES:
            break
        head_end += 1
    found = _summary_message(messages, head_end)
    if found is None:
        start = head_end
        log = task = ""
        digest = _EMPTY_DIGEST
    else:
        index, summary = found
        log, task = parse_summary(content_text(summary.get("content")))
        start = index + 1
        if start < len(messages) and is_ack(messages[start]):
            start += 1
        digest = summary_digest(summary)
    cut = _cut_index(messages, start, keep_tokens)
    raw_tokens = sum(message_tokens(message) for message in messages[start:])
    total_tokens = sum(message_tokens(message) for message in messages)
    return View(head_end, start, cut, log, task, digest, raw_tokens, total_tokens)


def _visible_role(message: dict[str, Any]) -> str | None:
    role = message.get("role")
    return role if role in VISIBLE_ROLES else None


def validate(messages: list[dict[str, Any]]) -> None:
    """Reject output that would not reach a provider as a valid OpenAI message sequence."""
    if not isinstance(messages, list) or not messages:
        raise HermesCompactionError("memshort produced an empty message list")
    for message in messages:
        if not isinstance(message, dict) or not message.get("role"):
            raise HermesCompactionError("memshort produced a message without a role")
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        if not calls:
            continue
        ids = [call.get("id") for call in calls if isinstance(call, dict)]
        if not ids:
            continue
        answered: list[Any] = []
        for follower in messages[index + 1 :]:
            if follower.get("role") != "tool":
                break
            answered.append(follower.get("tool_call_id"))
        missing = [call_id for call_id in ids if call_id not in answered]
        if missing and index + 1 + len(answered) < len(messages):
            raise HermesCompactionError(
                f"memshort split a tool-call group: {missing!r} has no following tool result"
            )


def _chunks(
    messages: list[dict[str, Any]], max_tokens: int
) -> list[list[dict[str, Any]]]:
    """Cut a pending slice into om-sized pieces at safe message boundaries where possible."""
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    used = 0
    for index, message in enumerate(messages):
        size = message_tokens(message)
        if current and used + size > max_tokens and _safe_cut(messages, index):
            chunks.append(current)
            current = []
            used = 0
        current.append(message)
        used += size
    if current:
        chunks.append(current)
    return chunks


class ObservationalCompactionEngine(ContextEngine):
    """The memshort context engine: transcript summaries are the authority; observations are the memory."""

    threshold_percent: float = 0.75
    # Hermes' preflight estimate gate keeps at least this many head/tail messages protected. This engine protects
    # only leading system/developer rows and owns its own tail cut, so set both to 0: otherwise a conversation of
    # ten or fewer large turns never reaches should_compress_preflight() at the 100k activation size.
    protect_first_n = 0
    protect_last_n = 0
    emit_automatic_compaction_status = True

    def __init__(
        self,
        *,
        bin_path: str | None = None,
        data_dir: str | Path | None = None,
        activate_tokens: int | None = None,
        midturn_tokens: int | None = None,
        keep_tokens: int | None = None,
        buffer_tokens: int | None = None,
        foreground_timeout: float | None = None,
        background_timeout: float | None = None,
        runner: Runner | None = None,
        enabled: bool | None = None,
    ) -> None:
        self._bin = bin_path or os.environ.get("MEMSHORT_COMPACT_BIN") or "memshort"
        self._data_dir = Path(data_dir).expanduser() if data_dir else None
        # A bad threshold turns the engine off with the reason, like a bad selection does in pi and Claude Code:
        # raising here would fail Hermes' plugin load instead of reporting it.
        self.config_error = ""
        try:
            limits = thresholds()
        except HermesCompactionError as err:
            self.config_error = str(err)
            logger.error("memshort: compaction is off: %s", err)
            limits = {key: default for key, (_, default) in THRESHOLDS.items()}
        self.activate_tokens = _positive_int(
            activate_tokens, limits["compact_after_turn_tokens"]
        )
        self.midturn_tokens = _positive_int(
            midturn_tokens, limits["compact_during_turn_tokens"]
        )
        self.keep_tokens = _positive_int(keep_tokens, limits["keep_recent_tokens"])
        self.buffer_tokens = _positive_int(
            buffer_tokens, limits["observe_every_tokens"]
        )
        self.foreground_timeout = _positive_float(
            foreground_timeout,
            _env_float(
                "MEMSHORT_OM_FOREGROUND_TIMEOUT_S", DEFAULT_FOREGROUND_TIMEOUT_S
            ),
        )
        self.background_timeout = _positive_float(
            background_timeout,
            _env_float(
                "MEMSHORT_OM_BACKGROUND_TIMEOUT_S", DEFAULT_BACKGROUND_TIMEOUT_S
            ),
        )
        self._runner: Runner | None = runner
        self._enabled = (
            (os.environ.get("MEMSHORT_COMPACT", "on").strip().lower() != "off")
            if enabled is None
            else enabled
        ) and not self.config_error
        self._instance_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._generation = 0
        self._session_id = ""
        self._observation: Observation | None = None
        self._inflight: threading.Thread | None = None
        self._inflight_generation = 0
        self._cancel = threading.Event()
        self._cached: tuple[int, int, bool] | None = None
        self._completed_turn = False
        self._failure_cooldown_until = 0.0
        self._noop_backoff_until = 0.0
        self._last_error = ""
        self._api_mode = ""
        self.model = ""
        self.provider = ""
        # Token state the host reads directly.
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.compression_count = 0
        self.model_thresholds = {}

    @property
    def name(self) -> str:
        return PLUGIN_NAME

    # ── host lifecycle ────────────────────────────────────────────────────────────────────────────
    def update_from_response(self, usage: dict[str, Any]) -> None:
        usage = usage if isinstance(usage, dict) else {}
        prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        total = int(usage.get("total_tokens") or (prompt + completion))
        self.last_prompt_tokens = prompt
        self.last_completion_tokens = completion
        self.last_total_tokens = total

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        self.model = model or ""
        self.provider = provider or ""
        self._api_mode = api_mode or ""
        self.context_length = (
            int(context_length)
            if isinstance(context_length, int) and context_length > 0
            else 0
        )
        percent = float(self.threshold_percent or 0.75)
        chosen = percent
        best = -1
        for key, value in (self.model_thresholds or {}).items():
            if isinstance(key, str) and key and key in self.model and len(key) > best:
                try:
                    chosen = float(value)
                except (TypeError, ValueError):
                    continue
                best = len(key)
        self.threshold_tokens = int(self.context_length * chosen)

    def _cancel_background(self) -> None:
        """Stop the old worker's paid work at a session boundary; a late result was already discarded."""
        with self._lock:
            self._cancel.set()
            self._cancel = threading.Event()

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        home = (
            kwargs.get("hermes_home")
            or os.environ.get("HERMES_HOME")
            or (Path.home() / ".hermes")
        )
        self._cancel_background()
        with self._lock:
            self._generation += 1
            self._session_id = session_id or ""
            self._observation = None
            self._cached = None
            self._completed_turn = False
            if home and self._data_dir is None:
                self._data_dir = (
                    Path(str(home)).expanduser() / "plugin-data" / PLUGIN_NAME
                )
            self._failure_cooldown_until = 0.0
            self._last_error = ""

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        self._cancel_background()
        with self._lock:
            self._generation += 1
            self._observation = None
            self._cached = None
            self._completed_turn = False

    def on_session_reset(self) -> None:
        self._cancel_background()
        with self._lock:
            self._generation += 1
            self._session_id = ""
            self._observation = None
            self._cached = None
            self._completed_turn = False
            self._failure_cooldown_until = 0.0
            self._last_error = ""
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            self.last_total_tokens = 0
            self.compression_count = 0

    def clone_for_agent(self) -> ObservationalCompactionEngine:
        """A fresh per-agent engine: no lock, thread or observation is shared, only settings and budget state."""
        clone = type(self)(
            bin_path=self._bin,
            data_dir=self._data_dir,
            activate_tokens=self.activate_tokens,
            midturn_tokens=self.midturn_tokens,
            keep_tokens=self.keep_tokens,
            buffer_tokens=self.buffer_tokens,
            foreground_timeout=self.foreground_timeout,
            background_timeout=self.background_timeout,
            runner=self._runner,
            enabled=self._enabled,
        )
        clone.model_thresholds = dict(self.model_thresholds or {})
        clone.threshold_percent = self.threshold_percent
        clone.context_length = self.context_length
        clone.threshold_tokens = self.threshold_tokens
        clone.last_prompt_tokens = self.last_prompt_tokens
        clone.last_completion_tokens = self.last_completion_tokens
        clone.last_total_tokens = self.last_total_tokens
        clone.compression_count = self.compression_count
        clone.model = self.model
        clone.provider = self.provider
        clone._api_mode = self._api_mode
        return clone

    # ── compaction trigger ────────────────────────────────────────────────────────────────────────
    @property
    def _app_server(self) -> bool:
        """Codex app-server owns its own thread and compaction; never summarize it here."""
        return self._api_mode.strip().lower() == "codex_app_server"

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        if (
            not self._enabled
            or self._app_server
            or time.monotonic() < self._noop_backoff_until
        ):
            return False
        with self._lock:
            cached = self._cached
        if cached is None:
            # No transcript snapshot yet; should_compress_preflight supplies one on this request.
            return False
        raw_tokens, total_tokens, compactable = cached
        if not compactable:
            return False
        if raw_tokens >= (
            self.activate_tokens if self._completed_turn else self.midturn_tokens
        ):
            return True
        # Safety net for a small model window whose fixed system/tools overhead makes the raw conversation
        # raw-token trigger too late to fit the request. A cold start has no prior prompt count, so the whole
        # message list (summary and head included) is the only pressure signal available.
        pressure = max(total_tokens, prompt_tokens or 0)
        return bool(self.threshold_tokens and pressure >= self.threshold_tokens)

    def should_compress_preflight(self, messages: list[dict[str, Any]]) -> bool:
        if (
            not self._enabled
            or self._app_server
            or time.monotonic() < self._noop_backoff_until
        ):
            return False
        view = analyze(messages, self.keep_tokens)
        self._remember(view)
        self._start_background(messages, view)
        threshold = (
            self.activate_tokens if self._completed_turn else self.midturn_tokens
        )
        if not view.compactable:
            return False
        if view.raw_tokens >= threshold:
            return True
        # A small model window can be near its limit while the raw conversation is below its activation threshold. Compare
        # every message (summary and head included) and the last real prompt with the window threshold: a cold
        # start has no usage yet, and a new large user turn can cross the window before any response reports it.
        pressure = max(view.total_tokens, self.last_prompt_tokens)
        return bool(self.threshold_tokens and pressure >= self.threshold_tokens)

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        return self._enabled and analyze(messages, self.keep_tokens).compactable

    # ── compaction ────────────────────────────────────────────────────────────────────────────────
    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        if not self._enabled or self._app_server:
            return messages
        if force:
            self._clear_compression_failure_cooldown()
        elif self.get_active_compression_failure_cooldown():
            # A recently failed pass is visible to the host; don't hammer memshort on every turn.
            return messages
        view = analyze(messages, self.keep_tokens)
        self._remember(view)
        if not view.compactable:
            return messages
        observation = self._matching_observation(view, messages)
        through = view.start
        log, task = view.log, view.task
        if (
            observation is not None
            and view.start + observation.covered_count <= view.cut
        ):
            through = view.start + observation.covered_count
            log, task = observation.log, observation.task
        pending = messages[through : view.cut]
        # A cold long session can hold far more than one om call's worth. Background observation bounds the
        # synchronous pass: defer instead of starting a CLI run that the 100 s host budget would abandon.
        if (
            not force
            and sum(message_tokens(message) for message in pending) > MAX_OBSERVE_TOKENS
            and self._start_background(messages, view)
        ):
            logger.info(
                "memshort: deferring compaction until background observation catches up"
            )
            return messages
        deadline = time.monotonic() + self.foreground_timeout
        cancel = self._cancel
        try:
            if pending:
                first = view.base_digest == _EMPTY_DIGEST and through == view.start
                log, task = self._observe(
                    log, task, pending, deadline, focus_topic, cancel, first
                )
            output = self._assemble(messages, view, log, task)
            validate(output)
        except HermesCompactionError as exc:
            self._record_compression_failure_cooldown(FAILURE_COOLDOWN_S, str(exc))
            raise
        except Exception as exc:
            error = HermesCompactionError(f"memshort compaction failed: {exc}")
            self._record_compression_failure_cooldown(FAILURE_COOLDOWN_S, str(error))
            raise error from exc
        self._clear_compression_failure_cooldown()
        self.compression_count += 1
        self._completed_turn = False
        with self._lock:
            self._observation = None
            self._cached = None
        self._remember(analyze(output, self.keep_tokens))
        return output

    def _assemble(
        self, messages: list[dict[str, Any]], view: View, log: str, task: str
    ) -> list[dict[str, Any]]:
        head = [copy.deepcopy(message) for message in messages[: view.head_end]]
        tail = [
            copy.deepcopy(message)
            for message in messages[view.cut :]
            if not is_summary(message) and not is_ack(message)
        ]
        output = [
            *head,
            {
                "role": "user",
                "content": render_summary(log, task),
            },
        ]
        # When the tail starts with a user, a fixed assistant acknowledgement prevents adjacent user rows.
        # When it starts with an assistant, the summary itself is the preceding user row.
        if tail and _visible_role(tail[0]) == "user":
            output.append({"role": "assistant", "content": ACK_CONTENT})
        output.extend(tail)
        return output

    # ── observation ──────────────────────────────────────────────────────────────────────────────
    def _observe(
        self,
        log: str,
        task: str,
        pending: list[dict[str, Any]],
        deadline: float,
        focus: str | None,
        cancel: threading.Event | None = None,
        first: bool = False,
    ) -> tuple[str, str]:
        """Observe a slice in order, one om call per chunk, inside one total deadline. `first` when the slice starts
        the session, so its first user message is kept verbatim."""
        for index, chunk in enumerate(_chunks(pending, MAX_OBSERVE_TOKENS)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HermesCompactionError(
                    f"memshort om exceeded its {self.foreground_timeout:.0f} s compaction budget"
                )
            request: dict[str, Any] = {
                "log": log,
                "task": task,
                "transcript": transcript_parts(chunk, first and index == 0),
            }
            if focus:
                request["focus"] = focus
            result = self._call_om(request, remaining, cancel=cancel)
            log = str(result.get("log") or "")
            task = str(result.get("task") or "")
        return log, task

    def _call_om(
        self,
        request: dict[str, Any],
        timeout: float,
        *,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        if cancel is not None and cancel.is_set():
            raise HermesCompactionError("memshort om cancelled at a session boundary")
        try:
            if self._runner is not None:
                result = self._runner(request, max(1.0, timeout))
            else:
                result = self._run_memshort_om(
                    request, max(1.0, timeout), cancel=cancel
                )
        except HermesCompactionError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise HermesCompactionError(
                f"memshort om timed out after {timeout:.0f} s"
            ) from exc
        except Exception as exc:
            raise HermesCompactionError(f"memshort om failed: {exc}") from exc
        if not isinstance(result, dict):
            raise HermesCompactionError("memshort om returned no JSON object")
        if result.get("error"):
            raise HermesCompactionError(f"memshort om failed: {result['error']}")
        if not isinstance(result.get("log"), str) or not isinstance(
            result.get("task"), str
        ):
            raise HermesCompactionError("memshort om returned no log/task strings")
        return result

    def _run_memshort_om(
        self,
        request: dict[str, Any],
        timeout: float,
        *,
        cancel: threading.Event | None = None,
    ) -> dict[str, Any]:
        command = shutil.which(self._bin) or self._bin
        try:
            process = subprocess.Popen(
                [command, "om"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=tempfile.gettempdir(),
                start_new_session=os.name == "posix",
            )
        except FileNotFoundError as exc:
            raise HermesCompactionError(
                f"the memshort command {self._bin!r} was not found on PATH; install it (uv tool install .)"
            ) from exc
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(request))
            process.stdin.close()
        except OSError:
            pass  # the child died before reading stdin; communicate() reports why
        deadline = time.monotonic() + timeout
        stdout = stderr = ""
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    self._kill_process_group(process)
                    raise HermesCompactionError(
                        "memshort om cancelled at a session boundary"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._kill_process_group(process)
                    raise HermesCompactionError(
                        f"memshort om timed out after {timeout:.0f} s; its process group was killed"
                    )
                try:
                    stdout, stderr = process.communicate(timeout=min(0.25, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except subprocess.TimeoutExpired as exc:  # the final poll raced the deadline
            self._kill_process_group(process)
            raise HermesCompactionError(
                f"memshort om timed out after {timeout:.0f} s; its process group was killed"
            ) from exc
        payload: dict[str, Any] = {}
        try:
            parsed = json.loads((stdout or "").strip() or "{}")
            payload = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            payload = {}
        reason = str(payload.get("error") or "").strip()
        if process.returncode != 0 or reason:
            detail = (
                reason
                or (stderr or "").strip()[-500:]
                or f"exited {process.returncode}"
            )
            raise HermesCompactionError(f"memshort om failed: {detail}")
        if not payload:
            raise HermesCompactionError("memshort om returned no JSON")
        return payload

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[str]) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.communicate(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    def select_context(
        self,
        request_messages: list[dict[str, Any]],
        *,
        conversation_messages: list[dict[str, Any]] | None = None,
        incoming_message: dict[str, Any] | None = None,
        budget_tokens: int = 0,
    ) -> None:
        # Called on every provider request, including tool-loop iterations. This hook observes only the durable
        # conversation; returning None leaves Hermes' request unchanged.
        if not self._enabled or self._app_server or conversation_messages is None:
            return
        view = analyze(conversation_messages, self.keep_tokens)
        self._remember(view)
        self._start_background(conversation_messages, view)

    def on_turn_complete(
        self,
        messages: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self._enabled or self._app_server:
            return
        if isinstance(usage, dict):
            self.update_from_response(usage)
        view = analyze(messages, self.keep_tokens)
        self._remember(view)
        self._completed_turn = view.raw_tokens >= self.activate_tokens
        self._start_background(messages, view)

    def _start_background(self, messages: list[dict[str, Any]], view: View) -> bool:
        """Start (or find) the background observation for this view; True when one covers the pending slice."""
        if not view.compactable:
            return False
        with self._lock:
            if not self._session_id or self._data_dir is None:
                return False
            if (
                self._inflight is not None
                and self._inflight_generation == self._generation
            ):
                return True
            session_id, instance_id, generation = (
                self._session_id,
                self._instance_id,
                self._generation,
            )
            cancel = self._cancel
            observation = self._matching_observation(view, messages)
            through = view.start
            log, task = view.log, view.task
            if (
                observation is not None
                and view.start + observation.covered_count <= view.cut
            ):
                through = view.start + observation.covered_count
                log, task = observation.log, observation.task
            pending = messages[through : view.cut]
            if (
                not pending
                or sum(message_tokens(message) for message in pending)
                < self.buffer_tokens
            ):
                return False
            base_digest = view.base_digest
            covered_count = view.cut - view.start
            through_digest = prefix_digest(messages[view.start : view.cut])
            thread = threading.Thread(
                target=self._background_worker,
                args=(
                    session_id,
                    instance_id,
                    generation,
                    base_digest,
                    through,
                    covered_count,
                    through_digest,
                    log,
                    task,
                    pending,
                    cancel,
                    base_digest == _EMPTY_DIGEST and through == view.start,
                ),
                name="memshort-observation",
                daemon=True,
            )
            self._inflight = thread
            self._inflight_generation = generation
        try:
            thread.start()
        except RuntimeError as exc:
            with self._lock:
                if self._inflight is thread:
                    self._inflight = None
            logger.warning("memshort: could not start background observation: %s", exc)
            return False
        return True

    def _background_worker(
        self,
        session_id: str,
        instance_id: str,
        generation: int,
        base_digest: str,
        through: int,
        covered_count: int,
        through_digest: str,
        log: str,
        task: str,
        pending: list[dict[str, Any]],
        cancel: threading.Event,
        first: bool = False,
    ) -> None:
        deadline = time.monotonic() + self.background_timeout
        started = time.monotonic()
        try:
            try:
                for index, chunk in enumerate(_chunks(pending, MAX_OBSERVE_TOKENS)):
                    if cancel.is_set():
                        logger.debug(
                            "memshort: background observation cancelled at a session boundary"
                        )
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise HermesCompactionError(
                            f"background observation exceeded {self.background_timeout:.0f} s"
                        )
                    result = self._call_om(
                        {
                            "log": log,
                            "task": task,
                            "transcript": transcript_parts(chunk, first and index == 0),
                        },
                        remaining,
                        cancel=cancel,
                    )
                    log = str(result.get("log") or "")
                    task = str(result.get("task") or "")
            except Exception as exc:  # noqa: BLE001 - background work never breaks the turn
                if cancel.is_set():
                    logger.debug("memshort: background observation cancelled: %s", exc)
                else:
                    logger.warning("memshort: background observation failed: %s", exc)
                return
            observation = Observation(
                session_id=session_id,
                instance_id=instance_id,
                generation=generation,
                base_digest=base_digest,
                covered_count=covered_count,
                through_digest=through_digest,
                log=log,
                task=task,
                updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            with self._lock:
                # A worker that outlived its session must not publish into the new one.
                if (
                    generation != self._generation
                    or session_id != self._session_id
                    or instance_id != self._instance_id
                ):
                    return
                self._observation = observation
                self._persist(observation)
            logger.debug(
                "memshort: background observation covered %d messages in %.1f s",
                covered_count,
                time.monotonic() - started,
            )
        finally:
            with self._lock:
                if self._inflight is threading.current_thread():
                    self._inflight = None

    def _observation_path(self, session_id: str) -> Path | None:
        if self._data_dir is None or not session_id:
            return None
        key = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
        return self._data_dir / "observations" / f"{key}.json"

    def _persist(self, observation: Observation) -> None:
        path = self._observation_path(observation.session_id)
        if path is None:
            return
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            # Session text: the directory and the file are private from the moment they exist.
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(observation.as_dict(), stream, ensure_ascii=False)
            os.replace(temporary, path)
        except OSError as exc:
            logger.warning(
                "memshort: could not persist background observation to %s: %s",
                path,
                exc,
            )
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _read_observation(self, session_id: str) -> Observation | None:
        path = self._observation_path(session_id)
        if path is None or not path.is_file():
            return None
        try:
            return Observation.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    def _matches(
        self, observation: Observation, view: View, messages: list[dict[str, Any]]
    ) -> bool:
        if observation.session_id != self._session_id:
            return False
        if (
            observation.instance_id == self._instance_id
            and observation.generation != self._generation
        ):
            return False
        if observation.base_digest != view.base_digest:
            return False
        end = view.start + observation.covered_count
        if observation.covered_count <= 0 or end > len(messages):
            return False
        return prefix_digest(messages[view.start : end]) == observation.through_digest

    def _matching_observation(
        self, view: View, messages: list[dict[str, Any]]
    ) -> Observation | None:
        with self._lock:
            memory = self._observation
        if memory is not None and self._matches(memory, view, messages):
            return memory
        stored = self._read_observation(self._session_id)
        return (
            stored
            if stored is not None and self._matches(stored, view, messages)
            else None
        )

    # ── failure/backoff surface (duck-typed by the host) ─────────────────────────────────────────
    def get_active_compression_failure_cooldown(
        self, *, refresh: bool = False
    ) -> dict[str, Any] | None:
        remaining = self._failure_cooldown_until - time.monotonic()
        if remaining <= 0:
            return None
        return {"remaining_seconds": remaining, "error": self._last_error}

    def _record_compression_failure_cooldown(
        self, seconds: float, error: str | None = None
    ) -> None:
        self._failure_cooldown_until = max(
            self._failure_cooldown_until, time.monotonic() + max(0.0, float(seconds))
        )
        self._last_error = error or ""

    def _clear_compression_failure_cooldown(self) -> None:
        self._failure_cooldown_until = 0.0
        self._last_error = ""

    def _record_structural_no_op(self, reason: str) -> None:
        logger.debug("memshort: compaction made no progress (%s)", reason)
        # The host arms its own anti-thrash state from this callback; a deferral is not a visible failure, so
        # stop automatic retries for a moment without claiming the summary provider is in cooldown.
        self._noop_backoff_until = time.monotonic() + 30.0

    # ── inspection helpers (status/tests; not part of the host contract) ─────────────────────────
    def _remember(self, view: View) -> None:
        with self._lock:
            self._cached = (view.raw_tokens, view.total_tokens, view.compactable)

    def wait_background(self, timeout: float = 5.0) -> bool:
        """Join the current background observation; compaction never calls this (it must not wait)."""
        with self._lock:
            thread = self._inflight
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    @property
    def observation(self) -> Observation | None:
        with self._lock:
            return replace(self._observation) if self._observation is not None else None
