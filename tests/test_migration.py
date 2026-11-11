import json

from memshort import install, install_hermes, installation


def test_legacy_pi_compactor_blocks_install_until_disabled(tmp_path, monkeypatch):
    old = tmp_path / "old"
    (old / "pi").mkdir(parents=True)
    (old / "pyproject.toml").write_text('[project]\nname = "aimem"\n')
    legacy = old / "pi/compact.ts"
    legacy.write_text('pi.on("session_before_compact", handler)')
    slot = tmp_path / "agent/extensions/memshort.ts"
    settings = slot.parent.parent / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"extensions": [str(legacy)]}))
    monkeypatch.setattr(
        install, "PI_LINKS", {"compact": (tmp_path / "new/pi/compact.ts", slot)}
    )
    config = tmp_path / "legacy.json"
    monkeypatch.setenv("MEMSHORT_INSTALL_CONFIG", str(config))
    wanted = install._Wanted(frozenset({"compact"}), frozenset({"pi"}))
    assert install._legacy_conflicts(wanted).outcome == "failed"
    config.write_text(json.dumps({"features": ["memory"], "harnesses": ["pi"]}))
    assert install._legacy_conflicts(wanted).outcome == "skipped"
    config.unlink()
    legacy.write_text("// compatibility stub, no compaction hook")
    assert install._legacy_conflicts(wanted).outcome == "skipped"


def test_legacy_claude_compactor_blocks_but_memory_only_is_safe(tmp_path, monkeypatch):
    register = tmp_path / "old/mod/hooks/register.ts"
    register.parent.mkdir(parents=True)
    register.write_text("registerCompact(on, options)")
    monkeypatch.setattr(install, "_claude_cli", lambda: "claude")
    monkeypatch.setattr(
        install,
        "_legacy_catalog",
        lambda _, *args: (
            [{"id": "aimem@aimem"}]
            if args == ("list",)
            else [{"name": "aimem", "path": str(tmp_path / "old")}]
        ),
    )
    monkeypatch.setenv("MEMSHORT_INSTALL_CONFIG", str(tmp_path / "missing"))
    monkeypatch.setenv(installation.ENV_PATH, str(tmp_path / "new.json"))
    wanted = install._Wanted(frozenset({"compact"}), frozenset({"claude"}))
    assert install._legacy_conflicts(wanted).outcome == "failed"
    register.write_text("// memory hooks only")
    assert install._legacy_conflicts(wanted).outcome == "skipped"
    register.unlink()
    assert install._legacy_conflicts(wanted).outcome == "failed"


def test_hermes_legacy_link_migrates_to_the_new_name(tmp_path, monkeypatch):
    old = tmp_path / "old"
    plugin = old / "hermes"
    plugin.mkdir(parents=True)
    (old / "pyproject.toml").write_text('[project]\nname = "aimem"\n')
    (plugin / "plugin.yaml").write_text("name: aimem\n")
    new = tmp_path / "new/hermes"
    new.mkdir(parents=True)
    (new / "plugin.yaml").write_text("name: memshort\n")
    (new / "__init__.py").write_text("")
    home = tmp_path / "home"
    monkeypatch.setenv(install_hermes.ENV_HOME, str(home))
    legacy = install_hermes.legacy_plugin_link()
    legacy.parent.mkdir(parents=True)
    legacy.symlink_to(plugin)
    state = home / "plugin-data/aimem/session.json"
    state.parent.mkdir(parents=True)
    state.write_text("unchanged")
    assert install_hermes._install_plugin(new, dry_run=True).outcome == "planned"
    assert legacy.resolve() == plugin.resolve()
    assert install_hermes._install_plugin(new, dry_run=False).outcome == "updated"
    assert install_hermes.plugin_link().resolve() == new.resolve()
    assert not legacy.exists() and not legacy.is_symlink()
    assert state.read_text() == "unchanged"
