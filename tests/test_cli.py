import json
import os
import subprocess
import sys


def cli(tmp_path, *args, stdin=""):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "MEMSHORT_COMPACT_INSTALL_CONFIG": str(tmp_path / "compact.json"),
        "MEMSHORT_INSTALL_CONFIG": str(tmp_path / "legacy.json"),
        "CLAUDE_CONFIG_DIR": str(tmp_path / "claude"),
        "HERMES_HOME": str(tmp_path / "hermes"),
        "CODEX_HOME": str(tmp_path / "codex"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    return subprocess.run(
        [sys.executable, "-c", "from memshort import main; main()", *args],
        env=env,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )


def test_cli_om_protocol_without_a_model_call(tmp_path):
    result = cli(
        tmp_path, "om", stdin=json.dumps({"log": "", "task": "", "transcript": ""})
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["summary"].startswith("<memshort-observations>")


def test_first_install_requires_explicit_harness_choice(tmp_path):
    result = cli(tmp_path, "install", "--dry-run")
    assert result.returncode != 0
    assert "--harness" in result.stderr
    assert not (tmp_path / "compact.json").exists()


def test_pi_install_and_uninstall_are_isolated(tmp_path):
    result = cli(tmp_path, "install", "--harness", "pi")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "compact.json").read_text())["harnesses"] == ["pi"]
    settings = tmp_path / ".pi/agent/settings.json"
    assert len(json.loads(settings.read_text())["extensions"]) == 1
    result = cli(tmp_path, "uninstall", "--harness", "pi")
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(settings.read_text())["extensions"] == []


def test_an_unknown_endpoint_fails_without_installing(tmp_path):
    result = cli(tmp_path, "install", "--harness", "pi", "--model", "nowhere/x")
    assert result.returncode != 0
    assert "unknown endpoint 'nowhere'" in result.stdout
    assert not (tmp_path / "compact.json").exists()
