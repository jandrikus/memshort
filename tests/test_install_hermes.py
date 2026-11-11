"""memshort in Hermes: the plugin link and the `context.engine` setting, and what must not be touched.

Every Hermes command is faked: `HERMES_HOME` points at a temporary directory and `hermes config` is a small state
machine, so no real Hermes config is read or written and no `hermes` process runs.
"""

import json
import os
import re
import subprocess

import pytest

from memshort import install_hermes


class FakeHermes:
    """The `hermes` CLI as the module sees it: `config get` and `config set` for `context.engine`."""

    def __init__(
        self,
        engine: str | None = "compressor",
        fail: tuple[str, ...] = (),
        get_output: str | None = None,
    ):
        self.engine = engine
        self.fail = fail
        self.get_output = (
            get_output  # a broken or unparsable `config get` answer, when set
        )
        self.calls: list[list[str]] = []

    def run(self, cmd):
        self.calls.append(cmd)
        action = " ".join(cmd[1:3])
        if action in self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        if action == "config get":
            if self.get_output is not None:
                return subprocess.CompletedProcess(cmd, 0, self.get_output, "")
            if self.engine is None:
                return subprocess.CompletedProcess(
                    cmd, 0, "Config key not set: context.engine", ""
                )
            return subprocess.CompletedProcess(cmd, 0, json.dumps(self.engine), "")
        if action == "config set":
            self.engine = cmd[4]
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def hermes(tmp_path, monkeypatch):
    """A temporary Hermes home with a plugin source laid out the way the repo's `hermes/` is."""
    monkeypatch.setenv(install_hermes.ENV_HOME, str(tmp_path / "hermes-home"))
    source = tmp_path / "repo" / "hermes"
    source.mkdir(parents=True)
    (source / install_hermes.MANIFEST).write_text(
        "name: memshort\nversion: 0.1.0\ndescription: compaction\n"
    )
    (source / "__init__.py").write_text("def register(ctx):\n    pass\n")
    fake = FakeHermes()
    monkeypatch.setattr(
        install_hermes.shutil, "which", lambda name: name if name == "hermes" else None
    )
    monkeypatch.setattr(install_hermes, "_run", fake.run)
    return source, fake


def outcome(steps, label):
    found = [step for step in steps if step.label == label]
    assert len(found) == 1, [step.label for step in steps]
    return found[0].outcome, found[0].detail


def test_install_links_the_plugin_and_selects_the_engine(hermes):
    source, fake = hermes
    steps = install_hermes.install(source=source)
    assert outcome(steps, "hermes plugin memshort")[0] == "done"
    assert outcome(steps, "Hermes context engine")[0] == "updated"
    assert install_hermes.plugin_link().is_symlink()
    assert os.readlink(install_hermes.plugin_link()) == str(source)
    assert fake.engine == "memshort"
    assert fake.calls[-1] == ["hermes", "config", "set", "context.engine", "memshort"]


def test_a_second_install_changes_nothing(hermes):
    source, fake = hermes
    install_hermes.install(source=source)
    fake.calls.clear()
    steps = install_hermes.install(source=source)
    assert [step.outcome for step in steps] == ["skipped", "skipped"]
    assert fake.calls == [["hermes", "config", "get", "context.engine", "--json"]]


def test_another_engine_is_not_replaced_without_consent(hermes):
    """`context.engine: lcm` is somebody else's setting: install the plugin, leave the setting, say how to switch."""
    source, fake = hermes
    fake.engine = "lcm"
    steps = install_hermes.install(source=source)
    state, detail = outcome(steps, "Hermes context engine")
    assert state == "skipped"
    assert "not replaced without explicit consent" in detail
    assert "hermes config set context.engine memshort" in detail
    assert fake.engine == "lcm"
    assert install_hermes.plugin_link().is_symlink()


def test_consent_replaces_the_engine(hermes):
    source, fake = hermes
    fake.engine = "lcm"
    steps = install_hermes.install(source=source, replace_engine=True)
    assert outcome(steps, "Hermes context engine")[0] == "updated"
    assert fake.engine == "memshort"


def test_a_foreign_plugin_directory_is_left_alone(hermes):
    source, fake = hermes
    plugin = install_hermes.plugin_link()
    plugin.mkdir(parents=True)
    (plugin / "someone-elses.txt").write_text("not memshort\n")
    steps = install_hermes.install(source=source)
    assert outcome(steps, "hermes plugin memshort")[0] == "failed"
    assert outcome(steps, "Hermes context engine")[0] == "skipped"
    assert (plugin / "someone-elses.txt").read_text() == "not memshort\n"
    assert fake.engine == "compressor"


def test_missing_plugin_source_fails(hermes):
    source, _fake = hermes
    (source / install_hermes.MANIFEST).unlink()
    steps = install_hermes.install(source=source)
    assert outcome(steps, "hermes plugin memshort")[0] == "failed"
    assert not install_hermes.plugin_link().exists()


def test_a_manifest_that_names_another_plugin_is_refused(hermes):
    source, _fake = hermes
    (source / install_hermes.MANIFEST).write_text("name: something-else\n")
    steps = install_hermes.install(source=source)
    state, detail = outcome(steps, "hermes plugin memshort")
    assert state == "failed"
    assert "something-else" in detail
    assert not install_hermes.plugin_link().exists()


def test_dry_run_writes_nothing(hermes):
    source, fake = hermes
    steps = install_hermes.install(source=source, dry_run=True)
    assert [step.outcome for step in steps] == ["planned", "planned"]
    assert not install_hermes.plugin_link().exists()
    assert fake.engine == "compressor"
    assert [call[1:3] for call in fake.calls] == [["config", "get"]]


def test_uninstall_restores_the_builtin_engine_and_removes_the_link(hermes):
    source, fake = hermes
    install_hermes.install(source=source)
    steps = install_hermes.uninstall(source=source)
    assert outcome(steps, "hermes plugin memshort")[0] == "done"
    assert outcome(steps, "Hermes context engine")[0] == "updated"
    assert not install_hermes.plugin_link().is_symlink()
    assert fake.engine == "compressor"


def test_uninstall_leaves_a_link_into_another_checkout(hermes, tmp_path):
    """A foreign link is not memshort's to remove, and its engine setting is not memshort's to change either."""
    source, fake = hermes
    other = tmp_path / "other-checkout"
    other.mkdir()
    install_hermes.plugins_dir().mkdir(parents=True)
    install_hermes.plugin_link().symlink_to(other)
    fake.engine = "memshort"
    steps = install_hermes.uninstall(source=source)
    assert outcome(steps, "hermes plugin memshort")[0] == "skipped"
    assert outcome(steps, "Hermes context engine")[0] == "skipped"
    assert install_hermes.plugin_link().is_symlink()
    assert fake.engine == "memshort"
    assert fake.calls == []


def test_uninstall_restores_the_engine_before_it_removes_the_link(hermes):
    """Order matters: the engine must be back on the built-in one before the plugin it names disappears."""
    source, fake = hermes
    install_hermes.install(source=source)
    fake.calls.clear()
    steps = install_hermes.uninstall(source=source)
    assert [step.label for step in steps] == [
        "Hermes context engine",
        "hermes plugin memshort",
    ]
    assert [call[1:3] for call in fake.calls] == [["config", "get"], ["config", "set"]]
    assert not install_hermes.plugin_link().is_symlink()


def test_a_failed_restore_keeps_the_plugin(hermes):
    """Removing the plugin while `context.engine` still names memshort would leave Hermes without an engine."""
    source, fake = hermes
    install_hermes.install(source=source)
    fake.fail = ("config set",)
    steps = install_hermes.uninstall(source=source)
    assert outcome(steps, "Hermes context engine")[0] == "failed"
    state, detail = outcome(steps, "hermes plugin memshort")
    assert state == "skipped"
    assert "must not lose its engine" in detail
    assert install_hermes.plugin_link().is_symlink()


def test_an_unreadable_engine_fails_the_install_and_sets_nothing(hermes):
    """A `config get` that fails or prints junk is not "unset": the engine stays as it is."""
    source, fake = hermes
    fake.fail = ("config get",)
    steps = install_hermes.install(source=source)
    assert outcome(steps, "Hermes context engine")[0] == "failed"
    assert fake.engine == "compressor"
    assert not any(call[1:3] == ["config", "set"] for call in fake.calls)


def test_unparsable_engine_output_fails_closed(hermes):
    source, fake = hermes
    fake.get_output = "Traceback (most recent call last): ..."
    steps = install_hermes.install(source=source)
    state, detail = outcome(steps, "Hermes context engine")
    assert state == "failed"
    assert "cannot read context.engine" in detail
    assert fake.engine == "compressor"


def test_a_failed_engine_read_keeps_the_plugin_on_uninstall(hermes):
    source, fake = hermes
    install_hermes.install(source=source)
    fake.fail = ("config get",)
    steps = install_hermes.uninstall(source=source)
    assert outcome(steps, "Hermes context engine")[0] == "failed"
    assert outcome(steps, "hermes plugin memshort")[0] == "skipped"
    assert install_hermes.plugin_link().is_symlink()


def test_an_unset_engine_is_unset_not_an_error(hermes):
    """`hermes config get` answers an unset key with a line, not JSON; that really is "nothing selected"."""
    source, fake = hermes
    fake.engine = None
    steps = install_hermes.install(source=source)
    assert outcome(steps, "Hermes context engine")[0] == "updated"
    assert fake.engine == "memshort"


def test_uninstall_leaves_an_engine_someone_else_selected(hermes):
    source, fake = hermes
    install_hermes.install(source=source)
    fake.engine = "lcm"
    steps = install_hermes.uninstall(source=source)
    assert outcome(steps, "Hermes context engine")[0] == "skipped"
    assert fake.engine == "lcm"


def test_without_the_hermes_cli_the_plugin_stays(hermes, monkeypatch):
    """Without the CLI the engine cannot be checked, so the link is not removed either."""
    source, _fake = hermes
    install_hermes.install(source=source)
    monkeypatch.setattr(install_hermes.shutil, "which", lambda name: None)
    steps = install_hermes.uninstall(source=source)
    state, detail = outcome(steps, "Hermes context engine")
    assert state == "failed"
    assert "not on PATH" in detail
    assert outcome(steps, "hermes plugin memshort")[0] == "skipped"
    assert install_hermes.plugin_link().is_symlink()


def test_the_plugin_folder_is_the_hermes_home_override(tmp_path, monkeypatch):
    monkeypatch.setenv(install_hermes.ENV_HOME, str(tmp_path / "elsewhere"))
    assert install_hermes.hermes_home() == tmp_path / "elsewhere"
    assert install_hermes.plugins_dir() == tmp_path / "elsewhere" / "plugins"
    assert (
        install_hermes.plugin_link() == tmp_path / "elsewhere" / "plugins" / "memshort"
    )
    monkeypatch.delenv(install_hermes.ENV_HOME)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert install_hermes.hermes_home() == tmp_path / "home" / ".hermes"


def test_engine_reads_a_key_hermes_says_is_unset(hermes):
    _source, fake = hermes
    fake.engine = None
    assert install_hermes.engine("hermes") is None


def test_engine_raises_when_the_command_fails(hermes):
    """A failed read must not pass for "unset": that is what would overwrite a configured engine."""
    _source, fake = hermes
    fake.fail = ("config get",)
    with pytest.raises(install_hermes.EngineError):
        install_hermes.engine("hermes")


def test_engine_raises_on_an_answer_it_cannot_read(hermes):
    _source, fake = hermes
    fake.get_output = "<html>gateway error</html>"
    with pytest.raises(install_hermes.EngineError):
        install_hermes.engine("hermes")


def test_a_missing_cli_fails_the_install_rather_than_saving_it(hermes, monkeypatch):
    """Without the CLI the engine cannot be selected, so `--harness hermes` cannot claim success."""
    source, _fake = hermes
    monkeypatch.setattr(install_hermes.shutil, "which", lambda name: None)
    steps = install_hermes.install(source=source)
    state, detail = outcome(steps, "Hermes context engine")
    assert state == "failed"
    assert "not on PATH" in detail


def test_the_repo_plugin_carries_the_name_the_installer_links():
    """The other half of the contract: `hermes/` must be the plugin this installer links and selects.

    The folder, `plugin.yaml`'s `name` and `ContextEngine.name` are one name; if the engine module renames itself, the
    install would link a folder Hermes never selects.
    """
    plugin = install_hermes.default_source()
    if not (plugin / install_hermes.MANIFEST).is_file():
        pytest.skip("the hermes/ plugin is not in this checkout")
    assert install_hermes.manifest_name(plugin) == install_hermes.PLUGIN_NAME
    engine_source = (plugin / "engine.py").read_text(encoding="utf-8")
    assert re.search(r'^PLUGIN_NAME\s*=\s*"memshort"$', engine_source, re.MULTILINE)


def test_a_legacy_engine_name_is_migrated(hermes):
    source, fake = hermes
    fake.engine = "aimem"
    steps = install_hermes.install(source=source)
    assert outcome(steps, "Hermes context engine")[0] == "updated"
    assert fake.engine == "memshort"
