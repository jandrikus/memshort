"""The Codex hooks: rollout reading, observation state, log injection and the hooks.json entries.

The rollout records here have the shapes Codex 0.167.1 wrote in a live session (response_item messages, function
calls and outputs, reasoning, a compacted record after Codex's own summary)."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from memshort import codex, install, installation, om

NOW = datetime(2026, 11, 11, 2, 0).astimezone()


def record(kind: str, payload: dict) -> str:
    return json.dumps(
        {"timestamp": "2026-11-11T01:00:00.000Z", "type": kind, "payload": payload}
    )


def message(role: str, text: str) -> str:
    part = "output_text" if role == "assistant" else "input_text"
    return record(
        "response_item",
        {"type": "message", "role": role, "content": [{"type": part, "text": text}]},
    )


def first_turn() -> list[str]:
    return [
        record("session_meta", {"id": "s1"}),
        message(
            "developer",
            "<skills_instructions>codex's own instructions</skills_instructions>",
        ),
        message(
            "user", "<environment_context>\n  <cwd>/tmp</cwd>\n</environment_context>"
        ),
        message("user", "Rename load_config to load_settings everywhere."),
        record(
            "response_item",
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "private"}],
            },
        ),
        record(
            "response_item",
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"rg load_config"}',
                "call_id": "c1",
            },
        ),
        record(
            "response_item",
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "src/a.py:3: load_config()",
            },
        ),
        record("event_msg", {"type": "token_count"}),
        message("assistant", "**CONTEXT CHECKPOINT COMPACTION** Codex's own summary"),
        record("compacted", {"message": "summary", "replacement_history": []}),
        message(
            "developer",
            "<memshort-observations>an earlier injection</memshort-observations>",
        ),
    ]


@pytest.fixture
def rollout(tmp_path) -> Path:
    path = tmp_path / "rollout-2026-11-11T01-00-00-s1.jsonl"
    path.write_text("\n".join(first_turn()) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def selected():
    installation.save(
        installation.Selection(harnesses=("codex",), model="openai/gpt-x")
    )


@pytest.fixture
def observer(monkeypatch):
    seen: list[str] = []

    def complete(role, prompt):
        seen.append(prompt)
        n = len(seen)
        return (
            f"<observations>\n* 🟡 observation {n}\n</observations>\n<current-task>\ntask {n}\n</current-task>",
            {"input": 10, "output": 5},
        )

    monkeypatch.setattr(om, "complete", complete)
    return seen


def event(rollout: Path, name: str = "PreCompact", **extra) -> dict:
    return {
        "session_id": "s1",
        "transcript_path": str(rollout),
        "hook_event_name": name,
        "cwd": "/tmp",
        **extra,
    }


def test_the_rollout_reads_as_conversation_without_codex_context_or_summary(rollout):
    lines, read = codex.read_rollout(rollout, 0)
    assert lines == [
        "[User]: Rename load_config to load_settings everywhere.",
        '[Tool call exec_command: {"cmd":"rg load_config"}]',
        "[Tool result]: src/a.py:3: load_config()",
    ]
    assert read == 11


def test_a_line_still_being_written_waits_for_the_next_read(rollout):
    with rollout.open("a", encoding="utf-8") as stream:
        stream.write(message("user", "half")[:20])
    lines, read = codex.read_rollout(rollout, 0)
    assert read == 11 and len(lines) == 3


def test_an_unknown_item_is_kept_as_json():
    line = codex.render_item(
        {"type": "web_search_call", "id": "x", "action": {"query": "codex hooks"}}
    )
    assert (
        line
        == '[web_search_call]: {"type": "web_search_call", "action": {"query": "codex hooks"}}'
    )


def test_pre_compact_observes_once_and_session_start_injects_the_log(
    rollout, selected, observer
):
    out, code = codex.hook("pre-compact", json.dumps(event(rollout)))
    assert (out, code) == ({}, 0)
    assert (
        len(observer) == 1
        and "Rename load_config" in observer[0]
        and "private" not in observer[0]
    )
    assert (
        "Codex's own summary" not in observer[0]
        and "an earlier injection" not in observer[0]
    )

    # Nothing new in the rollout: no second model call.
    codex.hook("pre-compact", json.dumps(event(rollout)))
    assert len(observer) == 1

    with rollout.open("a", encoding="utf-8") as stream:
        stream.write(message("user", "Also update the README.") + "\n")
    codex.hook("pre-compact", json.dumps(event(rollout)))
    assert (
        len(observer) == 2
        and "Rename load_config" not in observer[1]
        and "README" in observer[1]
    )

    out, code = codex.hook(
        "session-start", json.dumps(event(rollout, "SessionStart", source="compact"))
    )
    context = out["hookSpecificOutput"]["additionalContext"]
    assert out["hookSpecificOutput"]["hookEventName"] == "SessionStart" and code == 0
    assert (
        "observation 1" in context
        and "observation 2" in context
        and "task 2" in context
    )
    assert context.startswith("<memshort-observations>")

    # Only after a compaction, never at startup or resume.
    assert (
        codex.hook(
            "session-start", json.dumps(event(rollout, "SessionStart", source="resume"))
        )[0]
        == {}
    )


def test_the_state_file_is_private(rollout, selected, observer):
    codex.hook("pre-compact", json.dumps(event(rollout)))
    (state,) = codex.state_dir().glob("*.json")
    assert state.stat().st_mode & 0o777 == 0o600
    assert codex.state_dir().stat().st_mode & 0o777 == 0o700


def test_nothing_runs_unless_the_selection_names_codex(rollout, observer):
    assert codex.hook("pre-compact", json.dumps(event(rollout))) == ({}, 0)
    installation.save(installation.Selection(harnesses=("pi",), model="openai/gpt-x"))
    assert codex.hook("pre-compact", json.dumps(event(rollout))) == ({}, 0)
    assert observer == []


def test_a_subagent_hook_is_ignored(rollout, selected, observer):
    assert codex.hook("pre-compact", json.dumps(event(rollout, agent_id="a1"))) == (
        {},
        0,
    )
    assert observer == []


def test_a_failed_observation_warns_and_leaves_the_lines_for_the_next_compaction(
    rollout, selected, observer, monkeypatch
):
    def fail(role, prompt):
        raise om.ModelError("openai/gpt-x: no API key: set OPENAI_API_KEY")

    original = om.complete
    monkeypatch.setattr(om, "complete", fail)
    out, code = codex.hook("pre-compact", json.dumps(event(rollout)))
    assert code == 0
    assert out["systemMessage"].startswith(
        "memshort: observation failed, Codex's own summary stands in"
    )
    assert "no API key" in out["systemMessage"]

    monkeypatch.setattr(om, "complete", original)
    codex.hook("pre-compact", json.dumps(event(rollout)))
    assert "Rename load_config" in observer[0]


def test_stop_observes_only_past_its_threshold(
    rollout, selected, observer, monkeypatch
):
    monkeypatch.setenv("MEMSHORT_OBSERVE_EVERY_TOKENS", "100000")
    assert codex.hook("stop", json.dumps(event(rollout, "Stop"))) == ({}, 0)
    assert observer == []
    monkeypatch.setenv("MEMSHORT_OBSERVE_EVERY_TOKENS", "10")
    codex.hook("stop", json.dumps(event(rollout, "Stop")))
    assert len(observer) == 1


def test_a_long_backlog_is_observed_in_pieces(rollout, selected, observer, monkeypatch):
    monkeypatch.setattr(codex, "MAX_OBSERVE_TOKENS", 15)
    codex.hook("pre-compact", json.dumps(event(rollout)))
    assert len(observer) == 3


FOREIGN = {
    "description": "mine",
    "hooks": {
        "PreCompact": [
            {
                "matcher": "auto",
                "hooks": [{"type": "command", "command": "notify-send compacting"}],
            }
        ],
        "Stop": [{"hooks": [{"type": "command", "command": "/usr/bin/lint-check"}]}],
    },
}


def test_hooks_are_added_next_to_foreign_ones_and_taken_out_without_them():
    added = codex.with_ours(FOREIGN, binary="/opt/memshort")
    assert added["description"] == "mine"
    pre = added["hooks"]["PreCompact"]
    assert pre[0] == FOREIGN["hooks"]["PreCompact"][0]
    assert pre[1] == {
        "matcher": "auto|manual",
        "hooks": [
            {
                "type": "command",
                "command": "/opt/memshort codex-hook pre-compact",
                "timeout": 1800,
            }
        ],
    }
    start = added["hooks"]["SessionStart"][0]
    assert (
        start["matcher"] == "^compact$"
        and start["hooks"][0]["additionalContextLimit"] == 0
    )
    assert added["hooks"]["Stop"][1] == {
        "hooks": [
            {
                "type": "command",
                "command": "/opt/memshort codex-hook stop",
                "timeout": 1800,
                "async": True,
            }
        ]
    }
    # Adding twice replaces ours instead of doubling them.
    assert codex.with_ours(added, binary="/opt/memshort") == added
    assert codex.without_ours(added) == FOREIGN


def test_install_and_uninstall_wire_codex_and_keep_foreign_hooks(monkeypatch):
    monkeypatch.setattr(
        codex.shutil,
        "which",
        lambda name: "/opt/memshort" if name == "memshort" else None,
    )
    path = codex.hooks_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(FOREIGN), encoding="utf-8")
    want = install._Wanted(frozenset({"compact"}), frozenset({"codex"}))

    (plan,) = install._codex_steps(want, dry_run=True, held_back=False)
    assert plan.outcome == "planned" and json.loads(path.read_text()) == FOREIGN

    (step,) = install._codex_steps(want, dry_run=False, held_back=False)
    assert step.outcome == "done" and "/hooks" in step.detail
    assert "codex-hook pre-compact" in path.read_text()
    (again,) = install._codex_steps(want, dry_run=False, held_back=False)
    assert again.outcome == "skipped"

    (removed,) = install._codex_steps(want, dry_run=False, held_back=False, remove=True)
    assert removed.outcome == "done" and json.loads(path.read_text()) == FOREIGN


def test_a_deselected_codex_loses_its_hooks_on_install(monkeypatch):
    path = codex.hooks_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(codex.with_ours({})), encoding="utf-8")
    want = install._Wanted(frozenset({"compact"}), frozenset({"pi"}))
    (step,) = install._codex_steps(want, dry_run=False, held_back=False)
    assert step.outcome == "done" and json.loads(path.read_text()) == {"hooks": {}}


def test_a_hooks_file_that_is_not_an_object_fails_the_step():
    path = codex.hooks_path()
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")
    want = install._Wanted(frozenset({"compact"}), frozenset({"codex"}))
    (step,) = install._codex_steps(want, dry_run=False, held_back=False)
    assert step.outcome == "failed" and path.read_text() == "[]"
