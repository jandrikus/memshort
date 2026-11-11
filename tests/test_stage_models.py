"""Separate observer and reflector models, and the endpoints a host defines for them."""

import json
import os
import subprocess
import sys

import pytest

from memshort import backends, installation, om


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    for variable in (
        "MEMSHORT_MODEL",
        "MEMSHORT_OBSERVER_MODEL",
        "MEMSHORT_REFLECTOR_MODEL",
    ):
        monkeypatch.delenv(variable, raising=False)


def save(**extra):
    path = installation.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": 2, "features": ["compact"], "harnesses": ["pi"], **extra}
        ),
        encoding="utf-8",
    )
    return path


def test_both_stages_use_the_shared_model_unless_one_is_named():
    save(model="deepseek/deepseek-v4-pro")
    assert installation.stage_model("observer") == ("deepseek/deepseek-v4-pro", "model")
    assert installation.stage_model("reflector") == (
        "deepseek/deepseek-v4-pro",
        "model",
    )
    save(model="deepseek/deepseek-v4-pro", reflector_model="ollama/qwen3:8b")
    assert installation.stage_model("observer")[0] == "deepseek/deepseek-v4-pro"
    assert installation.stage_model("reflector") == ("ollama/qwen3:8b", "file")
    assert installation.load().as_dict()["reflector_model"] == "ollama/qwen3:8b"
    assert "observer_model" not in installation.load().as_dict()


def test_the_most_specific_setting_wins(monkeypatch):
    save(model="openai/file-shared", observer_model="openai/file-observer")
    monkeypatch.setenv("MEMSHORT_MODEL", "openai/env-shared")
    assert installation.stage_model("observer") == ("openai/file-observer", "file")
    assert installation.stage_model("reflector") == (
        "openai/env-shared",
        "MEMSHORT_MODEL",
    )
    monkeypatch.setenv("MEMSHORT_OBSERVER_MODEL", "pi/a/env-observer")
    assert installation.stage_model("observer") == (
        "pi/a/env-observer",
        "MEMSHORT_OBSERVER_MODEL",
    )


def test_the_old_shared_variable_is_gone(monkeypatch):
    monkeypatch.setenv("MEMSHORT_OM_MODEL", "openai/ignored")
    assert om.model("observer") == installation.DEFAULT_MODEL


@pytest.mark.parametrize(
    ("extra", "reason"),
    [
        ({"reflector_model": "nope"}, "reflector_model: model must look like"),
        ({"reflector_model": "nowhere/x"}, "unknown endpoint 'nowhere'"),
        ({"curator_model": "openai/b"}, "unknown keys: curator_model"),
        ({"endpoints": {"pi": {"api": "openai", "base_url": "http://x"}}}, "not 'pi'"),
        (
            {
                "endpoints": {
                    "lab": {"api": "openai", "base_url": "http://x", "api_key": "sk"}
                }
            },
            "endpoint lab: unknown keys api_key",
        ),
    ],
)
def test_a_bad_stage_model_is_an_error(extra, reason):
    save(**extra)
    with pytest.raises(installation.SelectionError, match=reason):
        installation.load()
    with pytest.raises(om.ModelError, match=reason):
        om.model("reflector")


def test_each_completion_runs_on_its_stage_model(monkeypatch):
    save(model="pi/a/shared", reflector_model="pi/b/reflector")
    monkeypatch.setattr(backends.shutil, "which", lambda name: "/usr/bin/pi")
    used = []

    def run(cmd, **kwargs):
        used.append(cmd[cmd.index("--model") + 1])
        event = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "stopReason": "stop",
                "content": [
                    {"type": "text", "text": "<observations>* 🟡 x</observations>"}
                ],
            },
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(event) + "\n", "")

    monkeypatch.setattr(backends.subprocess, "run", run)
    monkeypatch.setenv("MEMSHORT_CONDENSE_LOG_ABOVE_TOKENS", "1")
    om.run({"log": "* 🟡 a long enough log", "transcript": "user: hi"})
    assert used == ["a/shared", "b/reflector"]


def test_a_stage_model_change_keeps_everything_else():
    save(model="openai/shared", thresholds={"keep_recent_tokens": 20_000})
    changed = installation.with_model(installation.load(), "observer_model", "pi/b/obs")
    assert changed.as_dict()["observer_model"] == "pi/b/obs"
    assert changed.as_dict()["thresholds"] == {"keep_recent_tokens": 20_000}
    assert (
        "observer_model"
        not in installation.with_model(changed, "observer_model", None).as_dict()
    )
    with pytest.raises(installation.SelectionError, match="cannot be unset"):
        installation.with_model(changed, "model", None)
    again = installation.normalize(model="anthropic/new", base=changed)
    assert dict(again.stage_models) == {"observer": "pi/b/obs"}


def test_an_endpoint_is_defined_used_and_cannot_be_removed_while_in_use():
    save(model="deepseek/deepseek-flash")
    lab = backends.endpoint_of(
        "lab", {"api": "openai", "base_url": "http://127.0.0.1:8080/v1/"}
    )
    selection = installation.with_endpoint(installation.load(), "lab", lab)
    selection = installation.with_model(selection, "reflector_model", "lab/qwen3-8b")
    installation.save(selection)
    data = json.loads(installation.config_path().read_text())
    assert data["endpoints"] == {
        "lab": {"api": "openai", "base_url": "http://127.0.0.1:8080/v1"}
    }
    r = om.route("reflector")
    assert r.endpoint is not None
    assert (r.backend, r.model_id, r.endpoint.base()) == (
        "lab",
        "qwen3-8b",
        "http://127.0.0.1:8080/v1",
    )
    with pytest.raises(installation.SelectionError, match="unknown endpoint 'lab'"):
        installation.with_endpoint(installation.load(), "lab", None)


def test_a_host_endpoint_replaces_the_built_in_of_the_same_name():
    save(
        model="openai/gpt-x",
        endpoints={
            "openai": {
                "api": "openai",
                "base_url": "https://gateway.example/v1",
                "api_key_env": "GATEWAY_KEY",
            }
        },
    )
    r = om.route("observer")
    assert r.endpoint is not None
    assert r.endpoint.base() == "https://gateway.example/v1"
    assert r.endpoint.api_key_env == ("GATEWAY_KEY",)


def cli(tmp_path, *args):
    return subprocess.run(
        [sys.executable, "-c", "from memshort import main; main()", *args],
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
            "HERMES_HOME": str(tmp_path / "hermes"),
            "CODEX_HOME": str(tmp_path / "codex"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_install_saves_stage_models_and_endpoint_cli_adds_a_server(tmp_path):
    result = cli(
        tmp_path,
        "install",
        "--harness", "pi",
        "--model", "deepseek/deepseek-flash",
        "--reflector-model", "pi/anthropic/claude-haiku-4-5",
    )  # fmt: skip
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads(installation.config_path().read_text())
    assert saved["model"] == "deepseek/deepseek-flash"
    assert saved["reflector_model"] == "pi/anthropic/claude-haiku-4-5"
    added = cli(
        tmp_path, "endpoint", "add", "lab", "--api", "openai",
        "--base-url", "http://127.0.0.1:8080/v1", "--api-key-env", "LAB_KEY",
        "--extra-body", '{"chat_template_kwargs": {"enable_thinking": false}}',
    )  # fmt: skip
    assert added.returncode == 0, added.stderr
    assert (
        "lab: openai API at http://127.0.0.1:8080/v1, key from LAB_KEY" in added.stdout
    )
    assert (
        cli(tmp_path, "config", "set", "reflector_model", "lab/qwen3-8b").returncode
        == 0
    )
    shown = cli(tmp_path, "config")
    assert "reflector_model = lab/qwen3-8b (file)" in shown.stdout
    assert "observer_model = deepseek/deepseek-flash (model)" in shown.stdout
    refused = cli(tmp_path, "endpoint", "remove", "lab")
    assert refused.returncode != 0 and "unknown endpoint 'lab'" in refused.stderr
    key = cli(
        tmp_path,
        "endpoint",
        "add",
        "x",
        "--api",
        "openai",
        "--base-url",
        "http://h",
        "--api-key-env",
        "sk-123 secret",
    )
    assert key.returncode != 0 and "environment variables" in key.stderr
