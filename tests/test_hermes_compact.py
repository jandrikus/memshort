"""The Hermes context-engine plugin: observational compaction, state validation, failures and the manifest.

These tests never start Hermes, call a provider or run the real `memshort om`: the engine takes a runner, so every
model call is a fake protocol object; the default subprocess path is exercised with local shell scripts only. The real `agent.context_engine` ABC is imported from the local Hermes checkout when it is
present; otherwise a minimal stand-in is installed, so the suite runs on a host without Hermes as well.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, Unpack

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1] / "src" / "memshort" / "integrations"
HERMES_AGENT_SRC = os.environ.get("HERMES_AGENT_SRC", "").strip()


def _install_stub() -> None:
    for name in [
        name for name in sys.modules if name == "agent" or name.startswith("agent.")
    ]:
        del sys.modules[name]
    agent = types.ModuleType("agent")
    agent.__path__ = []  # type: ignore[attr-defined]
    engine = types.ModuleType("agent.context_engine")

    class ContextEngine:
        threshold_percent = 0.75
        emit_automatic_compaction_status = True

    engine.ContextEngine = ContextEngine  # type: ignore[attr-defined]
    agent.context_engine = engine  # type: ignore[attr-defined]
    sys.modules["agent"] = agent
    sys.modules["agent.context_engine"] = engine


def _prepare_context_engine_import() -> None:
    """Use the explicitly named Hermes checkout, else the reproducible stand-in. Never probes $HOME."""
    source = os.environ.get("HERMES_AGENT_SRC", "").strip()
    if not source:
        _install_stub()
        return
    try:
        if str(Path(source)) not in sys.path:
            sys.path.append(source)
        import agent.context_engine  # noqa: F401 - imported for its side effect (the real ABC)
    except Exception:  # noqa: BLE001 - a broken/foreign checkout falls back to the stand-in
        _install_stub()


_prepare_context_engine_import()

from memshort.integrations import hermes as hermes_plugin
from memshort.integrations.hermes.engine import (
    ACK_CONTENT,
    DEFAULT_ACTIVATE_TOKENS,
    DEFAULT_BACKGROUND_TIMEOUT_S,
    DEFAULT_BUFFER_TOKENS,
    DEFAULT_FOREGROUND_TIMEOUT_S,
    DEFAULT_KEEP_TOKENS,
    PLUGIN_NAME,
    SUMMARY_CLOSE,
    HermesCompactionError,
    ObservationalCompactionEngine,
    analyze,
    is_ack,
    is_summary,
    message_tokens,
    parse_summary,
    render_summary,
    tokens_of,
    validate,
)


class Limits(TypedDict, total=False):
    activate_tokens: int
    midturn_tokens: int
    keep_tokens: int
    buffer_tokens: int


SMALL: Limits = {"activate_tokens": 100, "keep_tokens": 40, "buffer_tokens": 40}
# A buffer no test run reaches accidentally; background work is exercised explicitly.
NO_BUFFER: Limits = {"activate_tokens": 100, "keep_tokens": 40, "buffer_tokens": 10**9}


def flat(transcript) -> str:
    """A request's transcript as one string: the text runs and the preserved items' content, in order."""
    if isinstance(transcript, str) or transcript is None:
        return transcript or ""
    return "\n".join(
        part["text"] if "text" in part else part["preserved"]["content"]
        for part in transcript
    )


class FakeOm:
    """A protocol fake for `memshort om`: records requests (the transcript flattened) and appends one visible line
    to the log."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.failure: Exception | None = None

    def __call__(self, request: dict, timeout: float) -> dict:
        request = {**request, "transcript": flat(request.get("transcript"))}
        self.requests.append(copy.deepcopy(request))
        if self.failure is not None:
            raise self.failure
        marker = f"* 🟡 observed {len(request.get('transcript') or '')} chars"
        log = (request.get("log") or "").strip()
        return {
            "log": f"{log}\n{marker}".strip(),
            "task": request.get("task") or "continue the task",
            "summary": "",
            "usage": {},
        }


class BlockingOm(FakeOm):
    """A background observation that has started (and recorded its request) but not finished."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, request: dict, timeout: float) -> dict:
        request = {**request, "transcript": flat(request.get("transcript"))}
        self.requests.append(copy.deepcopy(request))
        self.started.set()
        assert self.release.wait(5), "test did not release the background observation"
        return {
            "log": f"{request.get('log') or ''}\n* background".strip(),
            "task": "background",
            "usage": {},
        }


def message(role: str, name: str, chars: int = 400, **extra) -> dict:
    return {"role": role, "content": f"{name} " + "x" * chars, **extra}


def tool_call(call_id: str, name: str = "bash") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({"command": "ls"})},
            }
        ],
    }


def tool_result(call_id: str, name: str = "bash", chars: int = 2_000) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": "result " + "y" * chars,
    }


def engine(
    tmp_path: Path, om: Callable[[dict, float], dict], **settings: Unpack[Limits]
) -> ObservationalCompactionEngine:
    values: Limits = {
        "activate_tokens": 100,
        "keep_tokens": 40,
        "buffer_tokens": 40,
        **settings,
    }
    return ObservationalCompactionEngine(
        data_dir=tmp_path / "hermes-home" / "plugin-data" / PLUGIN_NAME,
        runner=om,
        foreground_timeout=5,
        background_timeout=5,
        **values,
    )


@pytest.fixture
def conversation() -> list[dict]:
    return [
        {"role": "system", "content": "system prompt"},
        message("user", "old-u", 800),
        message("assistant", "old-a", 800),
        message("user", "recent-u", 100),
        message("assistant", "recent-a", 100),
    ]


def test_manifest_and_register_describe_the_same_engine():
    manifest = yaml.safe_load(
        (ROOT / "hermes" / "plugin.yaml").read_text(encoding="utf-8")
    )
    assert manifest["name"] == PLUGIN_NAME == "memshort"
    for field in ("name", "version", "description"):
        assert manifest.get(field), f"plugin.yaml lacks {field}"
    assert not manifest.get("python_dependencies")
    assert not manifest.get("pip_dependencies")
    assert manifest.get("requires_hermes")
    assert "register_context_engine" in (ROOT / "hermes" / "__init__.py").read_text(
        encoding="utf-8"
    )

    manifest_engine = ObservationalCompactionEngine(runner=FakeOm())
    assert manifest_engine.name == manifest["name"]

    class Collector:
        engine = None

        def register_context_engine(self, instance):
            self.engine = instance

    collector = Collector()
    hermes_plugin.register(collector)
    assert isinstance(collector.engine, ObservationalCompactionEngine)
    assert collector.engine.name == "memshort"


class RealOm:
    """`memshort om`'s own `run`, in process, with the model replaced: each observer call writes one line naming
    the conversation it saw. Exercises the real log format, block placement and summary."""

    def __init__(self, monkeypatch) -> None:
        from memshort import om

        self.om = om
        self.prompts: list[str] = []

        def complete(role, prompt):
            self.prompts.append(prompt)
            conversation = prompt.rsplit("<conversation>\n", 1)[1]
            words = " ".join(conversation.split("\n</conversation>")[0].split()[:6])
            return (
                f"<observations>\n* 🟡 saw {words}\n</observations>\n<current-task>\nt\n</current-task>",
                {},
            )

        monkeypatch.setattr(om, "complete", complete)

    def __call__(self, request: dict, timeout: float) -> dict:
        return self.om.run(request)


def skill_view_call(call_id: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "skill_view", "arguments": json.dumps(args)},
            }
        ],
    }


def blocks(summary: str) -> list:
    from memshort import om

    return om.split_log(parse_summary(summary)[0])


def test_hermes_keeps_first_message_and_skill_in_order_across_compactions(
    tmp_path, monkeypatch
):
    body = "First line\n</observation-log>\n<current-task>\nx\n</current-task>\nLast line 😀"
    skill = json.dumps({"success": True, "name": "example", "content": body})
    messages = [
        {"role": "user", "content": "Original request"},
        {"role": "assistant", "content": "Looking first"},
        skill_view_call("s1", {"name": "example"}),
        {"role": "tool", "tool_call_id": "s1", "name": "skill_view", "content": skill},
        {"role": "assistant", "content": "Applied the skill"},
        {"role": "user", "content": "Follow up " + "x" * 400},
    ]
    made = engine(tmp_path, RealOm(monkeypatch), keep_tokens=1, buffer_tokens=10**9)
    first = made.compress(messages, force=True)
    segments = blocks(first[0]["content"])
    assert segments[0] == {"kind": "user", "content": "Original request"}
    assert "Looking first" in segments[1]
    assert segments[2] == {"kind": "skill", "name": "example", "content": body}
    assert "Applied the skill" in segments[3]
    assert parse_summary(first[0]["content"])[1] == "t"
    later = [
        *first,
        {"role": "assistant", "content": "More work " + "y" * 400},
        {"role": "user", "content": "Next " + "z" * 400},
    ]
    second = made.compress(later, force=True)
    again = blocks(second[0]["content"])
    # Earlier entries stay as they were; newer observations follow, after the last block.
    assert again[:3] == segments[:3]
    assert again[3].startswith(segments[3]) and "[Assistant]: More" in again[3]
    assert second[0]["content"].count(body) == 1


def test_hermes_keeps_slash_and_auto_loaded_skill_turns_where_they_were(
    tmp_path, monkeypatch
):
    slash = '[IMPORTANT: The user has invoked the "review" skill. The full skill content is loaded below.]\n\nRead carefully.\n'
    auto = '[IMPORTANT: The "deploy" skill is auto-loaded. Follow its instructions for this session.]\n\nDeploy safely.'
    messages = [
        {"role": "user", "content": "Original request"},
        {"role": "assistant", "content": "Ready"},
        {"role": "user", "content": slash},
        {"role": "assistant", "content": "Reviewed"},
        {"role": "user", "content": auto},
        {"role": "assistant", "content": "Deployed"},
        {"role": "user", "content": "Next " + "x" * 400},
    ]
    made = engine(tmp_path, RealOm(monkeypatch), keep_tokens=1, buffer_tokens=10**9)
    segments = blocks(made.compress(messages, force=True)[0]["content"])
    assert [
        s["content"] if isinstance(s, dict) else s.split("saw ")[1] for s in segments
    ] == [
        "Original request",
        "[Assistant]: Ready",
        slash,
        "[Assistant]: Reviewed",
        auto,
        "[Assistant]: Deployed",
    ]
    assert segments[2]["name"] == "review" and segments[4]["name"] == "deploy"


def test_hermes_does_not_keep_a_skill_reference_file(tmp_path, monkeypatch):
    reference = json.dumps(
        {"success": True, "name": "review", "content": "Reference material"}
    )
    messages = [
        {"role": "user", "content": "Original"},
        skill_view_call(
            "ref", {"name": "review", "file_path": "references/details.md"}
        ),
        {
            "role": "tool",
            "tool_call_id": "ref",
            "name": "skill_view",
            "content": reference,
        },
        {"role": "user", "content": "Next " + "x" * 400},
    ]
    made = engine(tmp_path, RealOm(monkeypatch), keep_tokens=1, buffer_tokens=10**9)
    segments = blocks(made.compress(messages, force=True)[0]["content"])
    assert [s for s in segments if isinstance(s, dict)] == [
        {"kind": "user", "content": "Original"}
    ]


def test_hermes_keeps_the_text_of_a_first_message_with_an_image(tmp_path, monkeypatch):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AA=="},
                },
            ],
        },
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "Next " + "x" * 400},
    ]
    om = RealOm(monkeypatch)
    made = engine(tmp_path, om, keep_tokens=1, buffer_tokens=10**9)
    segments = blocks(made.compress(messages, force=True)[0]["content"])
    assert segments[0] == {"kind": "user", "content": "Describe this"}
    assert "carried 1 image(s), not kept" in om.prompts[0]


def test_a_summary_without_blocks_still_parses_the_old_way():
    old = render_summary("a log", "a task")
    assert "<memshort-preserved>" not in old
    assert parse_summary(old) == ("a log", "a task")


def test_plugin_never_imports_memshort():
    path = sys.modules["memshort.integrations.hermes.engine"].__file__
    assert path is not None
    source = Path(path).read_text(encoding="utf-8")
    assert "import memshort" not in source
    assert "from memshort" not in source


def test_defaults_are_the_approved_observational_sizes():
    assert (DEFAULT_ACTIVATE_TOKENS, DEFAULT_KEEP_TOKENS, DEFAULT_BUFFER_TOKENS) == (
        100_000,
        15_000,
        10_000,
    )
    assert (
        DEFAULT_FOREGROUND_TIMEOUT_S == 100.0
    )  # inside Hermes' 120 s compression inactivity budget
    assert DEFAULT_FOREGROUND_TIMEOUT_S < 120.0
    assert (
        DEFAULT_BACKGROUND_TIMEOUT_S == 1_200.0
    )  # bounded by `memshort om`'s own two 300 s retries per stage
    made = ObservationalCompactionEngine(runner=FakeOm())
    assert (made.activate_tokens, made.keep_tokens, made.buffer_tokens) == (
        100_000,
        15_000,
        10_000,
    )
    assert (made.foreground_timeout, made.background_timeout) == (100.0, 1_200.0)


def test_completed_turn_and_midturn_have_separate_preflight_thresholds():
    om = FakeOm()
    made = ObservationalCompactionEngine(runner=om)
    messages = [
        {"role": "system", "content": "system prompt"},
        message("user", "very-old", 160_000),  # 40k raw tokens
        message("assistant", "old-answer", 160_000),
        message("user", "tail", 160_000),  # the newest 40k tokens: the kept tail
    ]
    assert made.should_compress() is False  # no snapshot yet
    assert made.should_compress_preflight(messages[:3]) is False
    assert made.should_compress_preflight(messages) is False  # 120k during a user turn
    made.on_turn_complete(messages)
    assert made.should_compress() is True  # the cached completed turn is over 100k
    assert made.should_compress_preflight(messages) is True
    assert (
        made.should_compress_preflight(messages) is True
    )  # a deferred compaction remains due
    longer = messages + [message("assistant", "more", 320_000)]
    fresh = ObservationalCompactionEngine(runner=FakeOm())
    assert fresh.should_compress_preflight(longer) is True  # 200k during a user turn
    view = analyze(messages, made.keep_tokens)
    assert view.raw_tokens >= 100_000
    assert view.cut > view.start  # something older than the kept tail


def test_long_single_user_turn_can_compact_between_tool_groups(tmp_path):
    om = FakeOm()
    made = engine(
        tmp_path,
        om,
        activate_tokens=100,
        midturn_tokens=200,
        keep_tokens=40,
        buffer_tokens=10**9,
    )
    messages = [
        {"role": "system", "content": "system prompt"},
        message("user", "task", 400),
        tool_call("one"),
        tool_result("one", chars=400),
        tool_call("two"),
        tool_result("two", chars=400),
    ]
    assert made.select_context(messages, conversation_messages=messages) is None
    assert made.should_compress() is True
    out = made.compress(messages, force=True)
    validate(out)
    assert out[1]["role"] == "user" and is_summary(out[1])
    assert out[2]["tool_calls"][0]["id"] == "two"
    assert out[3]["tool_call_id"] == "two"
    assert "task" in om.requests[0]["transcript"]


def test_compaction_replaces_old_turns_with_an_observation_summary(
    tmp_path, conversation
):
    om = FakeOm()
    made = engine(tmp_path, om, **SMALL)
    out = made.compress(conversation)

    assert out[0] == conversation[0]
    assert out[1]["role"] == "user" and out[1]["content"].startswith(
        "<memshort-observations>"
    )
    assert out[2] == {"role": "assistant", "content": ACK_CONTENT}
    assert out[3:] == conversation[3:]  # newest user + assistant stay raw
    assert (
        next(m for m in out if m.get("role") not in ("system", "developer"))["role"]
        == "user"
    )
    log, task = parse_summary(out[1]["content"])
    assert "observed" in log
    assert task == "continue the task"
    assert "old-u" in om.requests[0]["transcript"]
    assert "recent-u" not in om.requests[0]["transcript"]
    validate(out)
    assert made.compression_count == 1


def test_kept_tail_never_splits_an_assistant_tool_call_group(tmp_path):
    om = FakeOm()
    messages = [
        {"role": "system", "content": "system prompt"},
        message("user", "old-u", 800),
        message("assistant", "old-a", 800),
        message("user", "tool-u", 100),
        tool_call("call_1"),
        tool_result("call_1"),
        message("user", "next-u", 100),
        message("assistant", "done-a", 100),
    ]
    # The tool result is large enough that a naive token cut lands inside it; the engine must move to the user turn.
    out = engine(
        tmp_path, om, activate_tokens=100, keep_tokens=600, buffer_tokens=10**9
    ).compress(messages)
    assert out[1]["role"] == "user" and out[1]["content"].startswith(
        "<memshort-observations>"
    )
    assert out[2] == {"role": "assistant", "content": ACK_CONTENT}
    assert out[3]["content"].startswith("tool-u")
    assert out[4]["role"] == "assistant" and out[4].get("tool_calls")
    assert out[5]["role"] == "tool" and out[5]["tool_call_id"] == "call_1"
    validate(out)

    # With a small kept tail the whole group is observed instead of being cut apart.
    dropped = engine(
        tmp_path / "small-tail",
        FakeOm(),
        activate_tokens=100,
        keep_tokens=10,
        buffer_tokens=10**9,
    ).compress(messages)
    assert dropped[1]["content"].startswith("<memshort-observations>")
    assert not any(item.get("role") == "tool" for item in dropped)
    assert not any(item.get("tool_calls") for item in dropped)
    validate(dropped)


def test_compaction_failure_is_visible_and_preserves_the_originals(
    tmp_path, conversation
):
    om = FakeOm()
    om.failure = RuntimeError("deepseek is down")
    made = engine(tmp_path, om, **SMALL)
    before = copy.deepcopy(conversation)

    with pytest.raises(HermesCompactionError, match="deepseek is down"):
        made.compress(conversation)
    assert conversation == before
    cooldown = made.get_active_compression_failure_cooldown()
    assert cooldown and cooldown["remaining_seconds"] > 0

    # The failure is visible through the cooldown instead of retrying on every turn.
    assert made.compress(conversation) is conversation
    assert len(om.requests) == 1

    # A manual /compress forces a retry; it fails again and still keeps the originals.
    with pytest.raises(HermesCompactionError):
        made.compress(conversation, force=True)
    assert conversation == before
    assert len(om.requests) == 2


def test_om_timeout_is_a_visible_failure_inside_the_hermes_budget(
    tmp_path, conversation
):
    om = FakeOm()
    om.failure = subprocess.TimeoutExpired("memshort om", 1)
    made = engine(tmp_path, om, **SMALL)
    before = copy.deepcopy(conversation)
    with pytest.raises(HermesCompactionError, match="timed out"):
        made.compress(conversation)
    assert conversation == before


def test_repeated_compaction_uses_the_transcript_summary_as_authority(
    tmp_path, conversation
):
    om = FakeOm()
    made = engine(tmp_path, om, **NO_BUFFER)
    first = made.compress(conversation)
    first_log, _ = parse_summary(first[1]["content"])
    assert (
        made.compress(first, force=True) is first
    )  # nothing older than the kept tail: no model call
    assert len(om.requests) == 1

    later = [
        *first,
        message("user", "new-u", 400),
        message("assistant", "new-a", 400),
    ]
    second = made.compress(later, force=True)
    assert len(om.requests) == 2
    request = om.requests[1]
    assert request["log"].startswith(
        first_log
    )  # the previous log came from the transcript, not a side file
    assert "recent-u" in request["transcript"]
    assert (
        "old-u" not in request["transcript"]
    )  # already covered by the transcript summary
    summaries = [
        item for item in second if "<memshort-observations>" in item.get("content", "")
    ]
    assert len(summaries) == 1
    assert summaries[0]["role"] == "user"
    log, _ = parse_summary(summaries[0]["content"])
    assert log.count("observed") == 2
    assert second[-1]["content"].startswith("new-a")
    assert "new-u" in request["transcript"]
    assert second[-1]["content"].startswith("new-a")
    validate(second)


def _background_messages() -> list[dict]:
    return [
        {"role": "system", "content": "system prompt"},
        message("user", "u1", 400),
        message("assistant", "a1", 400),
        message("user", "u2", 400),
        message("assistant", "a2", 400),
        message("user", "u3", 400),
        message("assistant", "a3", 400),
    ]


def test_request_selection_starts_background_observation_during_a_turn(tmp_path):
    made = engine(
        tmp_path, FakeOm(), activate_tokens=300, keep_tokens=40, buffer_tokens=40
    )
    messages = _background_messages()
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    assert made.select_context(messages, conversation_messages=messages) is None
    assert made.wait_background(5)
    assert made.observation is not None


def test_background_observation_persists_and_a_resumed_engine_observes_only_the_delta(
    tmp_path,
):
    om = FakeOm()
    made = engine(tmp_path, om, activate_tokens=300, keep_tokens=40, buffer_tokens=40)
    first = _background_messages()
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    made.on_turn_complete(first)
    assert made.wait_background(5)
    assert made.observation is not None and made.observation.covered_count > 0
    files = list(
        (tmp_path / "hermes-home" / "plugin-data" / PLUGIN_NAME / "observations").glob(
            "*.json"
        )
    )
    assert len(files) == 1
    if os.name == "posix":
        assert files[0].stat().st_mode & 0o777 == 0o600
        assert files[0].parent.stat().st_mode & 0o777 == 0o700
    stored = json.loads(files[0].read_text(encoding="utf-8"))
    assert stored["session_id"] == "session-1"
    assert stored["through_digest"]
    assert "observed" in stored["log"]

    # A resumed agent (a new engine instance) trusts the digest-validated background log and only observes the delta.
    om2 = FakeOm()
    resumed = engine(
        tmp_path, om2, activate_tokens=300, keep_tokens=40, buffer_tokens=10**9
    )
    later = [
        *first,
        message("user", "u4", 400),
        message("assistant", "a4", 400),
        message("user", "u5", 400),
        message("assistant", "a5", 400),
    ]
    resumed.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    out = resumed.compress(later)
    assert len(om2.requests) == 1
    assert "observed" in om2.requests[0]["log"]  # the persisted background observation
    assert (
        "a3" in om2.requests[0]["transcript"] and "u4" in om2.requests[0]["transcript"]
    )
    assert "u1" not in om2.requests[0]["transcript"]  # already covered
    assert out[0] == later[0]
    validate(out)


def test_persisted_state_is_rejected_when_the_covered_prefix_changed(tmp_path):
    first = _background_messages()
    made = engine(
        tmp_path, FakeOm(), activate_tokens=300, keep_tokens=40, buffer_tokens=40
    )
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    made.on_turn_complete(first)
    assert made.wait_background(5)

    changed = copy.deepcopy(first)
    changed[2]["content"] += " changed"
    om = FakeOm()
    reader = engine(
        tmp_path, om, activate_tokens=300, keep_tokens=40, buffer_tokens=10**9
    )
    reader.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    reader.compress(changed)
    assert (
        "observed" not in om.requests[0]["log"]
    )  # stale prefix digest: start from the transcript's own summary
    assert "u1" in om.requests[0]["transcript"]


def test_stale_generation_is_rejected_within_the_same_engine(tmp_path):
    first = _background_messages()
    made = engine(
        tmp_path, FakeOm(), activate_tokens=300, keep_tokens=40, buffer_tokens=40
    )
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    made.on_turn_complete(first)
    assert made.wait_background(5)
    assert made.observation is not None

    made.on_session_reset()
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    om = FakeOm()
    made._runner = om  # type: ignore[attr-defined]
    made.compress(first)
    assert "observed" not in om.requests[0]["log"]
    assert "u1" in om.requests[0]["transcript"]


def test_a_worker_that_finishes_after_a_session_switch_writes_nothing(tmp_path):
    om = BlockingOm()
    made = engine(tmp_path, om, activate_tokens=50, keep_tokens=40, buffer_tokens=10)
    messages = _background_messages()
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    made.on_turn_complete(messages)
    assert om.started.wait(5)

    made.on_session_reset()
    made.on_session_start("session-2", hermes_home=str(tmp_path / "hermes-home"))
    om.release.set()
    assert made.wait_background(5)
    assert made.observation is None
    files = list(
        (tmp_path / "hermes-home" / "plugin-data" / PLUGIN_NAME / "observations").glob(
            "*.json"
        )
    )
    assert files == []
    assert len(om.requests) == 1


def test_codex_app_server_is_excluded(tmp_path, conversation):
    om = FakeOm()
    made = engine(tmp_path, om, **SMALL)
    made.update_model(
        "gpt-5", 200_000, provider="openai-codex", api_mode="codex_app_server"
    )
    assert made.should_compress_preflight(conversation) is False
    assert made.compress(conversation) is conversation
    assert om.requests == []


def test_clone_for_agent_does_not_share_observation_state(tmp_path):
    made = engine(tmp_path, FakeOm(), **SMALL)
    made.update_model("deepseek-flash", 200_000)
    made.compression_count = 3
    clone = made.clone_for_agent()
    assert clone is not made
    assert clone.threshold_tokens == made.threshold_tokens == 150_000
    assert clone.compression_count == 3
    assert clone._instance_id != made._instance_id  # type: ignore[attr-defined]
    assert clone.observation is None
    assert clone.activate_tokens == made.activate_tokens


def test_a_cold_long_session_defers_instead_of_abandoning_a_slow_cli_run(tmp_path):
    om = FakeOm()
    made = engine(tmp_path, om, activate_tokens=300, keep_tokens=40, buffer_tokens=40)
    made.on_session_start("cold", hermes_home=str(tmp_path / "hermes-home"))
    messages = [
        {"role": "system", "content": "system prompt"},
        *[
            item
            for index in range(1, 6)
            for item in (
                message("user", f"u{index}", 20_000),
                message("assistant", f"a{index}", 20_000),
            )
        ],
    ]
    before = copy.deepcopy(messages)

    # More than one 30k-token om call: compaction returns the originals and lets the background catch up.
    assert made.compress(messages) is messages
    assert messages == before
    assert made.wait_background(5)
    assert (
        len(om.requests) == 2
    )  # 40k pending -> two bounded chunks, observed in the background

    out = made.compress(messages)
    assert out is not messages
    assert (
        len(om.requests) == 2
    )  # the foreground pass only rendered the completed observation log
    assert out[0] == messages[0]
    validate(out)


def test_structural_no_op_backs_off_without_a_failure_cooldown(tmp_path, conversation):
    made = engine(
        tmp_path,
        FakeOm(),
        activate_tokens=100,
        keep_tokens=40,
        buffer_tokens=40,
        midturn_tokens=100,
    )
    assert made.should_compress_preflight(conversation) is True
    made._record_structural_no_op("deferred")  # type: ignore[attr-defined]
    assert made.should_compress_preflight(conversation) is False
    assert made.get_active_compression_failure_cooldown() is None
    made._noop_backoff_until = 0.0  # type: ignore[attr-defined]
    assert made.should_compress_preflight(conversation) is True


def test_preflight_protection_attrs_do_not_hide_short_large_sessions():
    made = ObservationalCompactionEngine(runner=FakeOm())
    assert (made.protect_first_n, made.protect_last_n) == (0, 0)
    assert (
        made.protect_first_n + made.protect_last_n + 1 == 1
    )  # Hermes' estimate gate no longer suppresses us


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "memshort"
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_default_backend_speaks_the_memshort_om_json_protocol(tmp_path):
    script = _script(
        tmp_path,
        """if [ "$1" != "om" ]; then exit 2; fi
cat >/dev/null
printf '%s' '{"log":"a log","task":"a task","summary":"unused","usage":{}}'
""",
    )
    made = ObservationalCompactionEngine(bin_path=str(script))
    assert made._run_memshort_om({"log": "", "task": "", "transcript": "hi"}, 5) == {
        "log": "a log",
        "task": "a task",
        "summary": "unused",
        "usage": {},
    }


def test_default_backend_surfaces_an_error_payload(tmp_path):
    script = _script(
        tmp_path,
        """printf '%s' '{"error":"MEMSHORT_OM_MODEL is not usable"}'
exit 1
""",
    )
    made = ObservationalCompactionEngine(bin_path=str(script))
    with pytest.raises(HermesCompactionError, match="MEMSHORT_OM_MODEL is not usable"):
        made._run_memshort_om({"log": "", "task": "", "transcript": "hi"}, 5)


def test_default_backend_kills_the_cli_process_group_on_timeout(tmp_path):
    script = _script(tmp_path, "sleep 30\n")
    made = ObservationalCompactionEngine(bin_path=str(script))
    started = time.monotonic()
    with pytest.raises(HermesCompactionError, match="timed out"):
        made._run_memshort_om({"log": "", "task": "", "transcript": "hi"}, 1)
    assert time.monotonic() - started < 5


def test_installer_manifest_name_and_source_match_the_plugin():
    install_hermes = pytest.importorskip("memshort.install_hermes")
    assert install_hermes.PLUGIN_NAME == PLUGIN_NAME
    assert install_hermes.manifest_name(ROOT / "hermes") == PLUGIN_NAME
    assert install_hermes.default_source().resolve() == (ROOT / "hermes").resolve()


def test_a_huge_old_turn_cannot_block_compaction(tmp_path):
    om = FakeOm()
    made = engine(
        tmp_path, om, activate_tokens=100, keep_tokens=40, buffer_tokens=10**9
    )
    messages = [
        {"role": "system", "content": "system prompt"},
        message("user", "huge-u", 20_000),
        message("assistant", "huge-a", 20_000),
        message("user", "recent-u", 100),
        message("assistant", "recent-a", 100),
    ]
    view = analyze(messages, made.keep_tokens)
    assert (
        view.cut == 3
    )  # not start: the old turn is observed instead of blocking every compaction
    out = made.compress(messages)
    assert out is not messages
    assert out[1]["role"] == "user" and out[1]["content"].startswith(
        "<memshort-observations>"
    )
    assert out[2] == {"role": "assistant", "content": ACK_CONTENT}
    assert out[3]["content"].startswith("recent-u")
    assert "huge-u" in om.requests[0]["transcript"]
    validate(out)


def test_rendered_summary_round_trips_and_has_a_real_closing_tag():
    text = render_summary("a log line", "a task")
    assert text.startswith("<memshort-observations>")
    assert text.endswith(SUMMARY_CLOSE)
    assert parse_summary(text) == ("a log line", "a task")
    assert is_summary({"role": "assistant", "content": text})
    assert not is_summary({"role": "user", "content": "<memshort-observations> pasted"})


def test_real_prompt_pressure_can_activate_before_50k_raw_on_a_small_window(tmp_path):
    made = engine(
        tmp_path, FakeOm(), activate_tokens=50_000, keep_tokens=40, buffer_tokens=10**9
    )
    made.update_model("small-model", 64_000)
    messages = [
        {"role": "system", "content": "system prompt"},
        message("user", "u1", 4_000),  # ~1k raw tokens
        message("assistant", "a1", 4_000),
        message("user", "u2", 4_000),
        message("assistant", "a2", 4_000),
    ]
    assert (
        made.should_compress_preflight(messages) is False
    )  # no pressure yet, raw is below 50k
    assert analyze(messages, made.keep_tokens).compactable
    made.update_from_response(
        {"prompt_tokens": 48_000, "completion_tokens": 10, "total_tokens": 48_010}
    )
    assert (
        made.should_compress_preflight(messages) is True
    )  # last real prompt hit 75% of a 64k window
    assert made.should_compress(48_000) is True


def test_repeated_compress_while_background_is_active_defers_without_a_foreground_call(
    tmp_path,
):
    om = BlockingOm()
    made = engine(tmp_path, om, activate_tokens=300, keep_tokens=40, buffer_tokens=40)
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    messages = [
        {"role": "system", "content": "system prompt"},
        *[
            item
            for index in range(1, 6)
            for item in (
                message("user", f"u{index}", 20_000),
                message("assistant", f"a{index}", 20_000),
            )
        ],
    ]
    made.on_turn_complete(messages)
    assert om.started.wait(5)
    active = len(om.requests)
    assert active == 1  # the background worker is mid-flight

    # Both attempts while the worker is active return the originals instead of starting a second huge
    # synchronous observation that the 100 s compression budget could not finish.
    assert made.compress(messages) is messages
    assert made.compress(messages) is messages
    assert len(om.requests) == active

    om.release.set()
    assert made.wait_background(5)
    background_calls = len(om.requests)
    out = made.compress(messages)
    assert out is not messages
    assert (
        len(om.requests) == background_calls
    )  # completed background log: no re-observation
    validate(out)


def test_token_estimate_uses_bytes_and_images_not_only_chars(tmp_path):
    assert tokens_of("x" * 400) == 100
    assert (
        tokens_of("Привет") > (len("Привет") + 3) // 4
    )  # Cyrillic is 2 bytes per character
    image = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}
        ],
    }
    assert message_tokens(image) >= 1_500


def test_default_home_data_dir_when_host_passes_no_kwargs(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    om = FakeOm()
    made = ObservationalCompactionEngine(
        runner=om, activate_tokens=50, keep_tokens=40, buffer_tokens=10
    )
    made.on_session_start("home-session")  # no hermes_home kwarg and no HERMES_HOME
    made.on_turn_complete(_background_messages())
    assert made.wait_background(5)
    files = list(
        (tmp_path / ".hermes" / "plugin-data" / "memshort" / "observations").glob(
            "*.json"
        )
    )
    assert len(files) == 1
    stored = json.loads(files[0].read_text(encoding="utf-8"))
    assert stored["session_id"] == "home-session"
    assert "observed" in stored["log"]


@pytest.mark.skipif(
    not HERMES_AGENT_SRC
    or not (Path(HERMES_AGENT_SRC) / "agent" / "context_engine.py").is_file(),
    reason="set HERMES_AGENT_SRC to a Hermes checkout to run the real discovery/lifecycle probe",
)
def test_real_hermes_discovers_loads_and_compacts_through_the_plugin(tmp_path):
    """Opt-in: the local Hermes runtime discovers the linked plugin, compacts with a fake `memshort`, and ends."""
    hermes = shutil.which("hermes")
    if hermes is None:
        pytest.skip("the hermes CLI is not on PATH")
    printed = subprocess.run(
        [hermes, "--print-runtime-command"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if printed.returncode != 0:
        pytest.skip(f"cannot read the Hermes runtime command: {printed.stderr[-300:]}")
    runtime = json.loads(printed.stdout)
    python = runtime[0] if isinstance(runtime, list) and runtime else None
    if not python or not Path(python).is_file():
        pytest.skip("the Hermes runtime interpreter is unavailable")

    fake = tmp_path / "bin" / "memshort"
    fake.parent.mkdir(parents=True)
    requests = tmp_path / "om-requests.jsonl"
    fake.write_text(
        f"""#!{sys.executable}
import json, os, sys
sys.path.insert(0, {str(ROOT.parents[1])!r})
from memshort import om
request = json.load(sys.stdin)
with open(os.environ["MEMSHORT_PROBE_REQUESTS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(request) + "\\n")
# memshort's own engine; only the model is replaced.
om.complete = lambda role, prompt: ("<observations>\\n* 🟡 observed slice\\n</observations>\\n<current-task>\\ncontinue\\n</current-task>", {{}})
print(json.dumps(om.run(request)))
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)

    home = tmp_path / "hermes-home"
    (home / "plugins").mkdir(parents=True)
    (home / "plugins" / "memshort").symlink_to(ROOT / "hermes")
    probe = f"""
import json, os, sys
sys.path.insert(0, {str(Path(HERMES_AGENT_SRC))!r})
sys._hermes_pin_default_home = True
import hermes_bootstrap
from plugins.context_engine import discover_context_engines, find_engine_dir, load_context_engine

found = [(name, available) for name, _, available in discover_context_engines() if name == "memshort"]
assert found == [("memshort", True)], found
assert find_engine_dir("memshort") is not None
engine = load_context_engine("memshort")
assert engine is not None and engine.name == "memshort", engine
assert (engine.activate_tokens, engine.midturn_tokens, engine.keep_tokens, engine.buffer_tokens) == (100, 200, 40, 1000000000), engine
engine.on_session_start("probe-session", hermes_home=os.environ["HERMES_HOME"], model="probe", context_length=100000)
messages = [
    {{"role": "system", "content": "system"}},
    {{"role": "user", "content": "old-u " + "x" * 800}},
    {{"role": "assistant", "content": "old-a " + "x" * 800, "tool_calls": [{{"id": "skill-1", "type": "function", "function": {{"name": "skill_view", "arguments": '{{"name":"example"}}'}}}}]}},
    {{"role": "tool", "name": "skill_view", "tool_call_id": "skill-1", "content": json.dumps({{"success": True, "name": "example", "content": "Verbatim skill instructions"}})}},
    {{"role": "user", "content": "recent-u " + "x" * 100}},
    {{"role": "assistant", "content": "recent-a " + "x" * 100}},
]
assert engine.should_compress_preflight(messages) is True
out = engine.compress(messages)
assert engine.compression_count == 1
assert [m["role"] for m in out] == ["system", "user", "assistant", "user", "assistant"], out
assert out[1]["content"].startswith("<memshort-observations>")
assert "observed slice" in out[1]["content"]
log = out[1]["content"].split("<observation-log>\\n", 1)[1]
first_at, skill_at, observed_at = (log.index(t) for t in ("old-u " + "x" * 800, "Verbatim skill instructions", "observed slice"))
assert first_at < observed_at < skill_at, log  # in conversation order: first message, its turn, the skill
assert out[2]["content"].startswith("Understood")
assert out[3]["content"].startswith("recent-u") and out[4]["content"].startswith("recent-a")
assert engine.compress(out, force=True) is out  # nothing older than the kept tail

later = [*out, {{"role": "user", "content": "new-u " + "x" * 800}}, {{"role": "assistant", "content": "new-a " + "x" * 800}}]
second = engine.compress(later, force=True)
assert engine.compression_count == 2
assert second[1]["role"] == "user" and second[1]["content"].startswith("<memshort-observations>")
assert second[1]["content"].count("observed slice") == 2
assert second[1]["content"].count("old-u " + "x" * 800) == 1
assert second[1]["content"].count("Verbatim skill instructions") == 1
assert second[-1]["content"].startswith("new-a")
engine.on_session_reset()
engine.on_session_end("probe-session", second)
print("MEMSHORT_HERMES_PROBE_OK")
"""
    env = {
        **os.environ,
        "HERMES_HOME": str(home),
        "MEMSHORT_COMPACT_BIN": str(fake),
        "MEMSHORT_PROBE_REQUESTS": str(requests),
        "MEMSHORT_COMPACT_AFTER_TURN_TOKENS": "100",
        "MEMSHORT_COMPACT_DURING_TURN_TOKENS": "200",
        "MEMSHORT_KEEP_RECENT_TOKENS": "40",
        "MEMSHORT_OBSERVE_EVERY_TOKENS": "1000000000",
        "MEMSHORT_OM_FOREGROUND_TIMEOUT_S": "10",
        "MEMSHORT_OM_BACKGROUND_TIMEOUT_S": "10",
    }
    env.pop("MEMSHORT_COMPACT", None)
    env.pop("MEMSHORT_OM_BIN", None)
    ran = subprocess.run(
        [python, "-I", "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr[-2_000:] + ran.stdout[-2_000:]
    assert "MEMSHORT_HERMES_PROBE_OK" in ran.stdout

    seen = [
        json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()
    ]
    assert len(seen) == 2, seen
    assert seen[0]["transcript"][0] == {
        "preserved": {"kind": "user", "content": "old-u " + "x" * 800}
    }
    assert seen[0]["transcript"][-1] == {
        "preserved": {
            "kind": "skill",
            "name": "example",
            "content": "Verbatim skill instructions",
        }
    }
    assert "recent-u" not in json.dumps(seen[0]["transcript"])
    # the real user turn survived the acknowledgement placement
    assert "recent-u" in json.dumps(seen[1]["transcript"])


@pytest.mark.parametrize("with_system", [True, False])
def test_summary_is_the_first_visible_user_turn_and_ack_is_skipped_next_read(
    tmp_path, with_system
):
    om = FakeOm()
    made = engine(tmp_path, om, **NO_BUFFER)
    messages = [
        *([{"role": "system", "content": "system prompt"}] if with_system else []),
        message("user", "old-u", 800),
        message("assistant", "old-a", 800),
        message("user", "recent-u", 100),
        message("assistant", "recent-a", 100),
    ]
    out = made.compress(messages)
    first_visible = next(m for m in out if m.get("role") not in ("system", "developer"))
    assert first_visible["role"] == "user"  # Anthropic/Bedrock requirement
    assert first_visible["content"].startswith("<memshort-observations>")
    summary_index = out.index(first_visible)
    assert is_ack(out[summary_index + 1])
    assert out[summary_index + 1]["content"] == ACK_CONTENT
    assert out[summary_index + 2]["content"].startswith("recent-u")

    # The acknowledgement and the summary are not raw conversation on the next read.
    view = analyze(out, made.keep_tokens)
    assert view.start == summary_index + 2
    assert view.log == parse_summary(first_visible["content"])[0]
    assert out[view.start]["content"].startswith("recent-u")


@pytest.mark.parametrize("boundary", ["reset", "end"])
def test_session_boundary_cancels_a_running_background_subprocess(tmp_path, boundary):
    started = tmp_path / "started"
    script = _script(tmp_path, f"touch {started}\nsleep 30\n")
    made = ObservationalCompactionEngine(
        bin_path=str(script),
        data_dir=tmp_path / "hermes-home" / "plugin-data" / PLUGIN_NAME,
        activate_tokens=50,
        keep_tokens=40,
        buffer_tokens=10,
        background_timeout=60,
    )
    made.on_session_start("session-1", hermes_home=str(tmp_path / "hermes-home"))
    messages = _background_messages()
    made.on_turn_complete(messages)
    deadline = time.monotonic() + 5
    while not started.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists(), "the background memshort om subprocess never started"

    if boundary == "reset":
        made.on_session_reset()
    else:
        made.on_session_end("session-1", messages)
    began = time.monotonic()
    assert made.wait_background(5)
    assert (
        time.monotonic() - began < 4
    )  # the subprocess group was killed, not left to its 60 s timeout
    files = list(
        (tmp_path / "hermes-home" / "plugin-data" / PLUGIN_NAME / "observations").glob(
            "*.json"
        )
    )
    assert files == []


def test_cold_start_small_window_uses_all_messages_before_any_usage(tmp_path):
    made = engine(
        tmp_path, FakeOm(), activate_tokens=50_000, keep_tokens=40, buffer_tokens=10**9
    )
    made.update_model("small-32k", 32_000)
    messages = [
        {"role": "system", "content": "system prompt " + "s" * 4_000},
        message("user", "old-u", 40_000),
        message("assistant", "old-a", 40_000),
        message("user", "new-u", 40_000),
    ]
    view = analyze(messages, made.keep_tokens)
    assert view.compactable
    assert view.raw_tokens < 50_000  # the 50k raw activation alone would not fire
    assert (
        view.total_tokens >= made.threshold_tokens
    )  # but the whole message list crosses 75% of 32k
    assert (
        made.last_prompt_tokens == 0
    )  # cold start: no prior provider usage to rely on

    assert made.should_compress_preflight(messages) is True
    made.update_from_response(
        {"prompt_tokens": 10_000, "completion_tokens": 1, "total_tokens": 10_001}
    )
    assert made.last_prompt_tokens < made.threshold_tokens
    assert (
        made.should_compress_preflight(messages) is True
    )  # prior prompt + the new large turn still crosses it


def test_cold_start_small_window_below_the_threshold_still_waits(tmp_path):
    made = engine(
        tmp_path, FakeOm(), activate_tokens=50_000, keep_tokens=40, buffer_tokens=10**9
    )
    made.update_model("small-32k", 32_000)
    messages = [
        {"role": "system", "content": "system prompt"},
        message("user", "old-u", 4_000),
        message("assistant", "old-a", 4_000),
        message("user", "new-u", 4_000),
    ]
    assert analyze(messages, made.keep_tokens).total_tokens < made.threshold_tokens
    assert made.should_compress_preflight(messages) is False


def test_a_summary_written_before_the_rename_is_still_recognized():
    old = "<aimem-observations>\n<observation-log>\nlog\n</observation-log>\n<current-task>\ntask\n</current-task>\n</aimem-observations>"
    assert is_summary({"role": "user", "content": old})
    assert parse_summary(old) == ("log", "task")


def test_hermes_reads_thresholds_from_the_selection_and_the_environment(
    tmp_path, monkeypatch
):
    path = tmp_path / "install.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "features": ["compact"],
                "harnesses": ["hermes"],
                "thresholds": {"compact_during_turn_tokens": 250_000},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMSHORT_COMPACT_INSTALL_CONFIG", str(path))
    monkeypatch.setenv("MEMSHORT_KEEP_RECENT_TOKENS", "20000")
    made = ObservationalCompactionEngine(runner=FakeOm())
    assert (made.activate_tokens, made.midturn_tokens, made.keep_tokens) == (
        100_000,
        250_000,
        20_000,
    )
    assert made.config_error == ""


@pytest.mark.parametrize(
    ("thresholds", "variable", "reason"),
    [
        ({"midturn": 1}, None, "unknown thresholds: midturn"),
        ({}, "1.5k", "MEMSHORT_OBSERVE_EVERY_TOKENS must be a positive"),
    ],
)
def test_a_bad_hermes_threshold_turns_compaction_off_with_the_reason(
    tmp_path, monkeypatch, thresholds, variable, reason
):
    path = tmp_path / "install.json"
    path.write_text(json.dumps({"thresholds": thresholds}), encoding="utf-8")
    monkeypatch.setenv("MEMSHORT_COMPACT_INSTALL_CONFIG", str(path))
    if variable:
        monkeypatch.setenv("MEMSHORT_OBSERVE_EVERY_TOKENS", variable)
    made = ObservationalCompactionEngine(runner=FakeOm())
    assert reason in made.config_error
    big = [message("user", "x", 2_000_000), message("assistant", "y", 400)]
    assert made.should_compress_preflight(big) is False
    assert made.compress(big) is big
