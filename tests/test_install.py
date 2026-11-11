"""Install into isolated fake harness homes, never the user's integrations."""

import json
import shutil
import subprocess

import pytest

from memshort import install, install_hermes, installation


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    package = tmp_path / "package" / "integrations"
    shutil.copytree(
        install.PACKAGE_ROOT,
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    repo = tmp_path / "installed" / "integrations"
    manifest = package / ".claude-plugin" / "marketplace.json"
    extensions = tmp_path / "pi" / "agent" / "extensions"
    monkeypatch.setattr(install, "REPO_ROOT", repo)
    monkeypatch.setattr(install, "PACKAGE_ROOT", package)
    monkeypatch.setattr(install, "MARKETPLACE_FILE", manifest)
    monkeypatch.setattr(
        install,
        "PI_LINKS",
        {"compact": (repo / "pi" / "compact.ts", extensions / "memshort.ts")},
    )
    monkeypatch.setenv(installation.ENV_PATH, str(tmp_path / "config" / "install.json"))
    monkeypatch.setenv(install_hermes.ENV_HOME, str(tmp_path / "hermes"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setattr(install, "_claude_cli", lambda: None)
    monkeypatch.setattr(install, "_json_list", lambda *args: [])
    monkeypatch.setattr(install, "_legacy_catalog", lambda *args: [])
    monkeypatch.setenv("MEMSHORT_INSTALL_CONFIG", str(tmp_path / "legacy-install.json"))
    return tmp_path


def failure(steps):
    return any(s.outcome in ("failed", "partial") for s in steps)


def test_install_pi_and_uninstall_preserve_other_extensions(sandbox):
    source, slot = install.PI_LINKS["compact"]
    settings = slot.parent.parent / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"extensions": ["~/other.ts"], "theme": "dark"}))
    selection = installation.config_path()
    plan = install.install(harnesses="pi", dry_run=True)
    assert not failure(plan)
    assert not selection.exists()
    assert json.loads(settings.read_text())["extensions"] == ["~/other.ts"]
    assert not failure(install.install(harnesses="pi"))
    saved = installation.load()
    assert saved.features == ("compact",) and saved.harnesses == ("pi",)
    data = json.loads(settings.read_text())
    assert data["theme"] == "dark" and data["extensions"][0] == "~/other.ts"
    assert (settings.parent / data["extensions"][1]).resolve() == source.resolve()
    assert not failure(install.install(harnesses="pi"))
    assert len(json.loads(settings.read_text())["extensions"]) == 2
    assert not failure(install.uninstall(harnesses="pi"))
    assert json.loads(settings.read_text()) == {
        "extensions": ["~/other.ts"],
        "theme": "dark",
    }
    assert selection.exists()


def test_invalid_pi_settings_fail_preflight_without_saving(sandbox):
    _, slot = install.PI_LINKS["compact"]
    settings = slot.parent.parent / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{bad")
    assert failure(install.install(harnesses="pi"))
    assert not installation.config_path().exists()
    assert settings.read_text() == "{bad"


def test_foreign_pi_slot_is_never_replaced(sandbox):
    _, slot = install.PI_LINKS["compact"]
    slot.parent.mkdir(parents=True)
    slot.write_text("not ours")
    assert failure(install.install(harnesses="pi"))
    assert not installation.config_path().exists()
    assert slot.read_text() == "not ours"


def test_owned_pi_symlink_is_migrated_to_settings(sandbox):
    source, slot = install.PI_LINKS["compact"]
    slot.parent.mkdir(parents=True)
    slot.symlink_to(source)
    assert not failure(install.install(harnesses="pi"))
    assert not slot.is_symlink()
    settings = slot.parent.parent / "settings.json"
    assert (
        settings.parent / json.loads(settings.read_text())["extensions"][0]
    ).resolve() == source.resolve()


def test_missing_marketplace_fails_before_any_action(sandbox, monkeypatch):
    install.MARKETPLACE_FILE.unlink()
    calls = []
    monkeypatch.setattr(install, "_claude_cli", lambda: "claude")
    monkeypatch.setattr(install, "_run", lambda cmd: calls.append(cmd))
    assert failure(install.install(harnesses="claude,pi"))
    assert not installation.config_path().exists()
    assert not calls


def test_execution_failure_does_not_wire_pi(sandbox, monkeypatch):
    monkeypatch.setattr(install, "_claude_cli", lambda: "claude")
    monkeypatch.setattr(install, "_marketplace_path", lambda _: None)
    monkeypatch.setattr(install, "_plugin_already_installed", lambda _: False)
    monkeypatch.setattr(
        install, "_run", lambda cmd: subprocess.CompletedProcess(cmd, 1, "", "failed")
    )
    steps = install.install(harnesses="claude,pi")
    assert failure(steps)
    assert installation.load().harnesses == ("claude", "pi")
    _, slot = install.PI_LINKS["compact"]
    assert not (slot.parent.parent / "settings.json").exists()


def test_first_install_carries_the_legacy_model(sandbox, monkeypatch):
    legacy = sandbox / "legacy-install.json"
    legacy.write_text(
        json.dumps(
            {"features": ["memory"], "harnesses": ["pi"], "model": "custom/model"}
        )
    )
    assert not failure(install.install(harnesses="pi"))
    # The memory product runs its model through pi, so it stays there.
    assert installation.load().model == "pi/custom/model"
    legacy.write_text("{broken")
    assert not failure(
        install.install(harnesses="pi")
    )  # the new selection now owns the choice


def test_invalid_legacy_model_requires_an_explicit_choice(sandbox):
    (sandbox / "legacy-install.json").write_text("{bad")
    assert failure(install.install(harnesses="pi"))
    assert not installation.config_path().exists()
    assert not failure(install.install(harnesses="pi", model="pi/custom/model"))


def test_invalid_existing_selection_is_never_replaced(sandbox):
    path = installation.config_path()
    path.parent.mkdir(parents=True)
    path.write_text('{"features":["memory"], "harnesses":["pi"]}')
    assert failure(install.install(harnesses="pi"))
    assert "memory" in path.read_text()


def test_uninstall_does_not_touch_foreign_legacy_harness(sandbox):
    _, slot = install.PI_LINKS["compact"]
    slot.parent.mkdir(parents=True)
    foreign = sandbox / "foreign.ts"
    foreign.write_text("unrelated")
    slot.symlink_to(foreign)
    assert not failure(install.uninstall(harnesses="pi"))
    assert slot.is_symlink() and slot.resolve() == foreign


def test_pi_direct_entry_resolves_like_pi_through_a_linked_agent_dir(sandbox):
    # A host whose ~/.pi is a link to a synced folder: pi resolves `../..` from the path as written, not from the
    # link's target. The installer must find the entry pi loads, not add a second one only this host can resolve.
    source, _ = install.PI_LINKS["compact"]
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("// compaction\n", encoding="utf-8")
    home = sandbox / "home"
    (sandbox / "synced" / "agent").mkdir(parents=True)
    home.mkdir()
    (home / ".pi").symlink_to(sandbox / "synced")
    (home / "checkout").symlink_to(source.parent)
    settings = home / ".pi" / "agent" / "settings.json"
    settings.write_text(
        json.dumps({"extensions": ["../../checkout/compact.ts"]}), encoding="utf-8"
    )
    install.PI_LINKS["compact"] = (
        source,
        home / ".pi" / "agent" / "extensions" / "memshort.ts",
    )

    assert install._link_pi_extension("compact", dry_run=True).outcome == "skipped"
    assert json.loads(settings.read_text(encoding="utf-8"))["extensions"] == [
        "../../checkout/compact.ts"
    ]
