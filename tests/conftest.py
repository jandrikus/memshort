import pytest

from memshort import installation


@pytest.fixture(autouse=True)
def isolated_selection(tmp_path, monkeypatch):
    """No test reads this host's selection or its threshold variables; a test that needs them sets its own."""
    monkeypatch.setenv(
        "MEMSHORT_COMPACT_INSTALL_CONFIG", str(tmp_path / "selection" / "install.json")
    )
    # Nor Codex's hooks file or memshort's Codex state.
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    for variable, _, _ in installation.THRESHOLDS.values():
        monkeypatch.delenv(variable, raising=False)
