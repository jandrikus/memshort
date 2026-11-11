"""`memshort doctor` on the pi route: pi and its credentials for a `pi/...` model, checked without a model call or a
secret. The direct endpoints are tested in test_backends.py."""

import json
import subprocess

import pytest

from memshort import doctor, om


@pytest.fixture(autouse=True)
def install_config(tmp_path, monkeypatch):
    """Keep the real install selection untouched, as in `test_om`. Every stage runs through pi here."""
    path = tmp_path / "config" / "aimem" / "install.json"
    monkeypatch.setenv("MEMSHORT_COMPACT_INSTALL_CONFIG", str(path))
    monkeypatch.delenv(om.ENV_MODEL, raising=False)
    save_selection(path, "pi/deepseek/deepseek-flash")
    return path


def save_selection(path, model):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": 2, "features": ["compact"], "harnesses": ["pi"], "model": model}
        ),
        encoding="utf-8",
    )


def fake_pi(monkeypatch, *runs, path="/usr/bin/pi"):
    """A pi that answers `--version` and `auth check` with `runs`, in order."""
    calls: list[list[str]] = []
    remaining = list(runs)

    def run(cmd, **kwargs):
        calls.append(cmd)
        return remaining.pop(0)

    monkeypatch.setattr(
        doctor.shutil, "which", lambda name: path if name == "pi" else None
    )
    monkeypatch.setattr(doctor.subprocess, "run", run)
    return calls


def completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def test_ready_reports_the_executable_and_the_credentials(monkeypatch):
    calls = fake_pi(
        monkeypatch,
        completed([], stdout="1.4.0\n"),
        completed(
            [], stdout='{"status":"ready","provider":"deepseek","authType":"api_key"}\n'
        ),
    )
    report = doctor.check()
    assert report.ok
    assert report.model == "pi/deepseek/deepseek-flash"
    assert [check.name for check in report.checks] == [
        "observer model",
        "reflector model",
        "pi executable",
        "model credentials",
    ]
    assert calls[1] == [
        "/usr/bin/pi",
        "auth",
        "check",
        "--model",
        "deepseek/deepseek-flash",
        "--json",
        "--no-refresh",
    ]
    out = report.render()
    assert "1.4.0" in out and "ready for deepseek" in out and out.endswith("ready")


def test_missing_pi_is_one_failed_check_and_no_subprocess(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)

    def no_run(*args, **kwargs):
        raise AssertionError("nothing should run without pi")

    monkeypatch.setattr(doctor.subprocess, "run", no_run)
    report = doctor.check()
    assert not report.ok
    assert "pi is not on PATH" in report.render()


def test_not_ready_credentials_are_reported_without_raw_output(monkeypatch):
    fake_pi(
        monkeypatch,
        completed([], stdout="1.4.0\n"),
        completed(
            [], returncode=1, stdout='{"status":"not_ready","provider":"deepseek"}\n'
        ),
    )
    report = doctor.check()
    assert not report.ok
    assert "not_ready for deepseek" in report.render()


def test_a_bad_selection_fails_before_credentials(install_config, monkeypatch):
    install_config.write_text("{not json", encoding="utf-8")
    fake_pi(monkeypatch, completed([], stdout="1.4.0\n"))
    report = doctor.check()
    assert not report.ok
    assert report.model is None
    assert "install selection" in report.render()
    # Nothing to run pi for: no model resolved.
    assert [check.name for check in report.checks] == [
        "observer model",
        "reflector model",
    ]


def test_auth_output_without_json_never_leaks_its_text(monkeypatch):
    fake_pi(
        monkeypatch,
        completed([], stdout="1.4.0\n"),
        completed([], stdout="sk-super-secret-value\n"),
    )
    report = doctor.check()
    assert not report.ok
    assert "sk-super-secret-value" not in report.render()
    assert "returned no JSON" in report.render()


def test_the_model_can_be_checked_without_changing_the_selection(
    install_config, monkeypatch
):
    calls = fake_pi(
        monkeypatch,
        completed([], stdout="1.4.0\n"),
        completed(
            [], stdout='{"status":"ready","provider":"openai","authType":"oauth"}\n'
        ),
    )
    before = install_config.read_text()
    report = doctor.check(model="pi/openai/gpt-5.4")
    assert report.ok
    assert report.model == "pi/openai/gpt-5.4"
    assert "openai/gpt-5.4" in calls[1]
    assert install_config.read_text() == before


def test_refresh_is_an_explicit_opt_in(monkeypatch):
    calls = fake_pi(
        monkeypatch,
        completed([], stdout="1.4.0\n"),
        completed(
            [], stdout='{"status":"ready","provider":"deepseek","authType":"api_key"}\n'
        ),
    )
    assert doctor.check(refresh=True).ok
    assert "--no-refresh" not in calls[1]


def test_an_unexpected_auth_status_is_not_ready(monkeypatch):
    fake_pi(
        monkeypatch,
        completed([], stdout="1.4.0\n"),
        completed([], stdout='{"status":"weird"}\n'),
    )
    report = doctor.check()
    assert not report.ok
    assert "unexpected pi auth status" in report.render()


def test_an_os_error_is_a_failed_check(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/pi")

    def fail(*args, **kwargs):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(doctor.subprocess, "run", fail)
    report = doctor.check()
    assert not report.ok
    assert "Permission denied" in report.render()
