"""Compaction thresholds: one table in four places, the file and environment precedence, and `memshort config`."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from memshort import installation, om

INTEGRATIONS = Path(installation.__file__).resolve().parent / "integrations"


def ts_table(path: Path) -> dict[str, tuple[str, int]]:
    text = path.read_text(encoding="utf-8")
    block = text[text.index("THRESHOLDS = {") : text.index("} as const")]
    rows = re.findall(r"(\w+): \[['\"](\w+)['\"], ([\d_]+)\]", block)
    return {
        key: (variable, int(value.replace("_", ""))) for key, variable, value in rows
    }


def test_every_adapter_carries_the_same_table():
    want = {
        key: (variable, default)
        for key, (variable, default, _) in installation.THRESHOLDS.items()
    }
    assert ts_table(INTEGRATIONS / "pi/lib/installation.ts") == want
    assert ts_table(INTEGRATIONS / "claude/hooks/installation.ts") == want
    source = (INTEGRATIONS / "hermes/engine.py").read_text(encoding="utf-8")
    block = source[source.index("THRESHOLDS: dict") : source.index("\n}\n")]
    rows = re.findall(r'"(\w+)": \("(\w+)", ([\d_]+)\)', block)
    assert {
        key: (variable, int(value.replace("_", ""))) for key, variable, value in rows
    } == want


def test_names_say_what_they_do_and_match_between_file_and_environment():
    for key, (variable, _, _) in installation.THRESHOLDS.items():
        assert variable == f"MEMSHORT_{key.upper()}"
        assert key.endswith("_tokens")


def save(path: Path, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": 2, "features": ["compact"], "harnesses": ["pi"], **extra}
        ),
        encoding="utf-8",
    )


def test_environment_wins_over_file_wins_over_default(monkeypatch):
    path = installation.config_path()
    save(path, thresholds={"compact_during_turn_tokens": 250_000})
    assert installation.threshold("compact_after_turn_tokens") == (100_000, "default")
    assert installation.threshold("compact_during_turn_tokens") == (250_000, "file")
    monkeypatch.setenv("MEMSHORT_COMPACT_DURING_TURN_TOKENS", "300_000")
    assert installation.threshold("compact_during_turn_tokens") == (
        300_000,
        "MEMSHORT_COMPACT_DURING_TURN_TOKENS",
    )


@pytest.mark.parametrize(
    ("thresholds", "reason"),
    [
        ({"activate": 1}, "unknown thresholds: activate"),
        ({"keep_recent_tokens": 0}, "keep_recent_tokens must be a positive"),
        ({"keep_recent_tokens": "lots"}, "keep_recent_tokens must be a positive"),
        ({"keep_recent_tokens": True}, "keep_recent_tokens must be a positive"),
        ([], "thresholds must be an object"),
    ],
)
def test_a_bad_file_threshold_is_an_error_not_a_default(thresholds, reason):
    save(installation.config_path(), thresholds=thresholds)
    with pytest.raises(installation.SelectionError, match=reason):
        installation.load()


def test_a_bad_environment_threshold_names_the_variable(monkeypatch):
    monkeypatch.setenv("MEMSHORT_KEEP_RECENT_TOKENS", "-5")
    with pytest.raises(
        installation.SelectionError, match="MEMSHORT_KEEP_RECENT_TOKENS"
    ):
        installation.threshold("keep_recent_tokens")


def test_install_keeps_the_thresholds_it_did_not_change():
    path = installation.config_path()
    save(
        path,
        model="deepseek/deepseek-flash",
        thresholds={"observe_every_tokens": 12_000},
    )
    again = installation.normalize(model="openai/gpt-5", base=installation.load())
    assert dict(again.thresholds) == {"observe_every_tokens": 12_000}
    assert again.as_dict()["thresholds"] == {"observe_every_tokens": 12_000}
    assert "thresholds" not in installation.Selection().as_dict()


def test_the_reflector_uses_the_configured_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        om,
        "complete",
        lambda role, prompt: (
            calls.append(role),
            ("<observations>short</observations>", {}),
        )[1],
    )
    save(installation.config_path(), thresholds={"condense_log_above_tokens": 10})
    assert om.run({"log": "x" * 100, "transcript": ""})["reflected"]
    assert calls == ["reflector"]


def cli(*args):
    return subprocess.run(
        [sys.executable, "-c", "from memshort import main; main()", "config", *args],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


def test_config_shows_sets_and_unsets_one_threshold():
    path = installation.config_path()
    assert cli("set", "keep_recent_tokens", "20000").returncode != 0  # no selection yet
    save(path, model="deepseek/deepseek-flash")
    shown = cli()
    assert shown.returncode == 0, shown.stderr
    assert "compact_during_turn_tokens = 200000 (default)" in shown.stdout
    assert "MEMSHORT_COMPACT_DURING_TURN_TOKENS" in shown.stdout
    changed = cli("set", "compact_during_turn_tokens", "250000")
    assert changed.returncode == 0, changed.stderr
    assert "compact_during_turn_tokens = 250000 (file)" in changed.stdout
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["thresholds"] == {"compact_during_turn_tokens": 250_000}
    assert data["model"] == "deepseek/deepseek-flash"
    assert data["version"] == 2
    refused = cli("set", "midturn", "1")
    assert refused.returncode != 0 and "unknown key 'midturn'" in refused.stderr
    assert cli("set", "keep_recent_tokens", "0").returncode != 0
    assert cli("unset", "compact_during_turn_tokens").returncode == 0
    assert "thresholds" not in json.loads(path.read_text(encoding="utf-8"))
