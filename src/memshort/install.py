"""Install independently packaged compaction adapters without changing memshort's memory integrations.

The already-installed Python package supplies the command and adapter files. The installer copies adapters to a
stable host-local path, validates and saves a selection, then wires the requested harnesses. A preflight refuses an
active pre-split compactor. Hermes' owned old plugin link is migrated while its engine name and data stay the same.
A later failure keeps the selection and reports a partial install. Uninstall leaves data and selection intact.
"""

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from memshort import codex, install_hermes, installation

# Detail shared by the steps an earlier failure holds back: a failed run must not tear anything down.
HELD_BACK = "left in place: an earlier step failed"


@dataclass
class Step:
    label: str
    outcome: str  # "done", "skipped", "updated", "failed", "planned"
    detail: str = ""


@dataclass(frozen=True)
class _Wanted:
    """The names a run acts on: a whole `Selection` for an install, a filter for an uninstall."""

    features: frozenset[str]
    harnesses: frozenset[str]

    def wants(self, name: str) -> bool:
        return name in self.features or name in self.harnesses


PACKAGE_ROOT = Path(__file__).resolve().parent / "integrations"
# A stable host-local path. Shared harness settings must not name a Python-version-specific uv tool environment.
REPO_ROOT = Path.home() / ".local/share/memshort/integrations"
MARKETPLACE_FILE = REPO_ROOT / ".claude-plugin/marketplace.json"
PI_EXTENSIONS = Path.home() / ".pi/agent/extensions"
# Feature -> (its file in the repo, its legacy Pi link). The Claude Code mod carries both and gates them at runtime.
PI_LINKS = {
    "compact": (REPO_ROOT / "pi/compact.ts", PI_EXTENSIONS / "memshort.ts"),
}
MARKETPLACE_NAME = "memshort"
PLUGIN_NAME = "memshort"
# The two steps the Hermes install reports.
HERMES_LABELS = (f"hermes plugin {install_hermes.PLUGIN_NAME}", "Hermes context engine")


def _marketplace_manifest() -> dict:
    return {
        "name": MARKETPLACE_NAME,
        "description": "Observational compaction for agent sessions",
        "owner": {"name": "memshort"},
        "plugins": [
            {
                "name": PLUGIN_NAME,
                "description": "Observational compaction for Claude Code",
                "version": "0.1.0",
                "source": "./claude",
            }
        ],
    }


def _same_repo(path: str | Path) -> bool:
    """Whether `path` names this checkout, reached through a home symlink or resolved, or is empty."""
    if not str(path):
        return False
    try:
        return Path(path).resolve() == Path(REPO_ROOT).resolve()
    except OSError:
        return False


def _link_target(link: Path) -> Path | None:
    """Where a symlink points, resolved as far as it exists; None when `link` is not a symlink."""
    if not link.is_symlink():
        return None
    target = Path(os.readlink(link))
    if not target.is_absolute():
        target = link.parent / target
    try:
        return target.resolve()
    except OSError:
        return target


def _owned_link(feature: str) -> bool:
    """Whether the pi extension slot of `feature` holds memshort's own link into this checkout."""
    source, link = PI_LINKS[feature]
    return _link_target(link) == Path(source).resolve()


# Only these packaged files are deployed. Never copy __pycache__ or unrelated files from an editable checkout.
RESOURCE_FILES = (
    ".claude-plugin/marketplace.json",
    "claude/.claude-plugin/plugin.json",
    "claude/hooks/hooks.json",
    "claude/hooks/register.ts",
    "claude/hooks/installation.ts",
    "claude/hooks/compact.tsx",
    "claude/hooks/preserved.ts",
    "claude/types/index.d.ts",
    "pi/compact.ts",
    "pi/lib/installation.ts",
    "hermes/__init__.py",
    "hermes/engine.py",
    "hermes/plugin.yaml",
)


def _install_resources(dry_run: bool) -> Step:
    label = "integration files"
    changes: list[tuple[Path, bytes]] = []
    try:
        for name in RESOURCE_FILES:
            source, target = PACKAGE_ROOT / name, REPO_ROOT / name
            data = source.read_bytes()
            for parent in (target, *target.parents):
                if parent == REPO_ROOT.parent:
                    break
                if parent.is_symlink():
                    raise ValueError(f"refusing to replace a symlink: {parent}")
            if not target.is_file() or target.read_bytes() != data:
                changes.append((target, data))
        if not changes:
            return Step(label, "skipped", f"up to date in {REPO_ROOT}")
        if dry_run:
            return Step(
                label, "planned", f"would copy {len(changes)} files to {REPO_ROOT}"
            )
        for target, data in changes:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
                temp = Path(tmp.name)
                try:
                    tmp.write(data)
                    tmp.flush()
                    os.chmod(temp, 0o644)
                    os.replace(temp, target)
                finally:
                    temp.unlink(missing_ok=True)
        return Step(label, "done", f"copied {len(changes)} files to {REPO_ROOT}")
    except (OSError, ValueError) as err:
        return Step(label, "failed", f"cannot deploy integrations: {err}")


def _write_marketplace(dry_run: bool = False) -> Step:
    label = "marketplace manifest"
    want = _marketplace_manifest()
    current = None
    packaged = PACKAGE_ROOT / ".claude-plugin/marketplace.json"
    if packaged.exists():
        try:
            current = json.loads(packaged.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = None
    if current == want:
        return Step(label, "skipped", str(MARKETPLACE_FILE))
    return Step(
        label,
        "failed",
        f"missing or invalid packaged marketplace manifest: {MARKETPLACE_FILE}",
    )


def _claude_cli() -> str | None:
    return shutil.which("claude")


def _claude_config_dir() -> Path:
    """Where Claude Code keeps its state: `$CLAUDE_CONFIG_DIR` when set, else `~/.claude`."""
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(override).expanduser() if override else Path.home() / ".claude"


def _claude_trace() -> list[Path]:
    """What a memshort Claude Code install leaves on disk, for when the CLI cannot be asked about it.

    Claude Code records installed plugins in `plugins/installed_plugins.json`, keeps each installed plugin under
    `plugins/cache/<marketplace>/<plugin>/`, and its marketplaces in `plugins/known_marketplaces.json`. Any of those
    naming memshort means there is something to remove here, so a missing `claude` CLI is a real blocker and not the
    empty host a pi-only machine is.
    """
    plugins = _claude_config_dir() / "plugins"
    found = [
        path
        for path in (
            plugins / "cache" / MARKETPLACE_NAME,
            plugins / "marketplaces" / MARKETPLACE_NAME,
        )
        if path.exists()
    ]
    installed, known = (
        plugins / "installed_plugins.json",
        plugins / "known_marketplaces.json",
    )
    try:
        records = (
            json.loads(installed.read_text(encoding="utf-8"))
            if installed.is_file()
            else {}
        )
        if isinstance(records, dict) and f"{PLUGIN_NAME}@{MARKETPLACE_NAME}" in (
            records.get("plugins") or {}
        ):
            found.append(installed)
    except (OSError, json.JSONDecodeError):
        pass
    try:
        marketplaces = (
            json.loads(known.read_text(encoding="utf-8")) if known.is_file() else {}
        )
        if MARKETPLACE_NAME in marketplaces:
            found.append(known)
    except (OSError, TypeError, json.JSONDecodeError):
        pass
    return found


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _json_list(claude: str, *args: str) -> list[dict]:
    probe = _run([claude, "plugin", *args, "--json"])
    if probe.returncode != 0:
        return []
    try:
        data = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return []
    return (
        [item for item in data if isinstance(item, dict)]
        if isinstance(data, list)
        else []
    )


def _marketplace_path(claude: str) -> str | None:
    """The directory the `memshort` marketplace points at, or None if it is not added."""
    for item in _json_list(claude, "marketplace", "list"):
        if item.get("name") == MARKETPLACE_NAME:
            return item.get("path") or ""
    return None


def _plugin_already_installed(claude: str) -> bool:
    want = f"{PLUGIN_NAME}@{MARKETPLACE_NAME}"
    return any(item.get("id") == want for item in _json_list(claude, "list"))


def _add_marketplace(claude: str, dry_run: bool = False) -> Step:
    label = "Claude Code marketplace"
    path = _marketplace_path(claude)
    if path is not None and _same_repo(path):
        return Step(label, "skipped", f"{MARKETPLACE_NAME} already added")
    if dry_run:
        detail = (
            f"would add {REPO_ROOT}"
            if path is None
            else f"would move from {path} to {REPO_ROOT}"
        )
        return Step(label, "planned", detail)
    # Adding it again from another path (the repo moved) repoints the marketplace and its installed mod.
    ran = _run(
        [claude, "plugin", "marketplace", "add", str(REPO_ROOT), "--scope", "user"]
    )
    if ran.returncode != 0:
        return Step(label, "failed", (ran.stderr or ran.stdout).strip()[-300:])
    if path is None:
        return Step(label, "done", f"added {REPO_ROOT}")
    return Step(label, "updated", f"moved from {path} to {REPO_ROOT}")


def _install_plugin(claude: str, update: bool = False, dry_run: bool = False) -> Step:
    label = "Claude Code mod"
    already = _plugin_already_installed(claude)
    if already and not update:
        return Step(
            label, "skipped", f"{PLUGIN_NAME}@{MARKETPLACE_NAME} already installed"
        )
    if dry_run:
        what = "would refresh" if already else "would install"
        return Step(label, "planned", f"{what} {PLUGIN_NAME}@{MARKETPLACE_NAME}")
    if already:
        ran = _run([claude, "plugin", "marketplace", "update", MARKETPLACE_NAME])
        if ran.returncode != 0:
            return Step(label, "failed", (ran.stderr or ran.stdout).strip()[-300:])
        return Step(label, "updated", f"refreshed {PLUGIN_NAME}@{MARKETPLACE_NAME}")
    ran = _run(
        [
            claude,
            "plugin",
            "install",
            f"{PLUGIN_NAME}@{MARKETPLACE_NAME}",
            "--scope",
            "user",
            "-y",
        ]
    )
    if ran.returncode != 0:
        return Step(label, "failed", (ran.stderr or ran.stdout).strip()[-300:])
    return Step(label, "done", f"installed {PLUGIN_NAME}@{MARKETPLACE_NAME}")


def _remove_plugin(claude: str, dry_run: bool = False) -> Step:
    label = "Claude Code mod"
    if not _plugin_already_installed(claude):
        return Step(
            label, "skipped", f"{PLUGIN_NAME}@{MARKETPLACE_NAME} is not installed"
        )
    if dry_run:
        return Step(
            label, "planned", f"would uninstall {PLUGIN_NAME}@{MARKETPLACE_NAME}"
        )
    # `--keep-data` keeps whatever the mod stored under ~/.claude/plugins/data; memshort keeps its data elsewhere.
    ran = _run(
        [
            claude,
            "plugin",
            "uninstall",
            f"{PLUGIN_NAME}@{MARKETPLACE_NAME}",
            "--scope",
            "user",
            "--keep-data",
            "-y",
        ]
    )
    if ran.returncode != 0:
        return Step(label, "failed", (ran.stderr or ran.stdout).strip()[-300:])
    return Step(label, "done", f"uninstalled {PLUGIN_NAME}@{MARKETPLACE_NAME}")


def _remove_marketplace(claude: str, dry_run: bool = False) -> Step:
    label = "Claude Code marketplace"
    path = _marketplace_path(claude)
    if path is None:
        return Step(label, "skipped", f"{MARKETPLACE_NAME} is not added")
    if not _same_repo(path):
        return Step(
            label,
            "skipped",
            f"left in place: {MARKETPLACE_NAME} points at {path}, not {REPO_ROOT}",
        )
    if dry_run:
        return Step(
            label, "planned", f"would remove {MARKETPLACE_NAME} from {REPO_ROOT}"
        )
    ran = _run(
        [claude, "plugin", "marketplace", "remove", MARKETPLACE_NAME, "--scope", "user"]
    )
    if ran.returncode != 0:
        return Step(label, "failed", (ran.stderr or ran.stdout).strip()[-300:])
    return Step(label, "done", f"removed {MARKETPLACE_NAME}")


def _unwire_claude(
    claude: str, dry_run: bool = False, held_back: bool = False
) -> list[Step]:
    """Take the memshort plugin and marketplace back, but only while the marketplace is still this repo's."""
    if held_back:
        return [
            Step("Claude Code mod", "skipped", HELD_BACK),
            Step("Claude Code marketplace", "skipped", HELD_BACK),
        ]
    path = _marketplace_path(claude)
    if path is not None and not _same_repo(path):
        why = f"left in place: the {MARKETPLACE_NAME} marketplace points at {path}, not {REPO_ROOT}"
        return [
            Step("Claude Code mod", "skipped", why),
            Step("Claude Code marketplace", "skipped", why),
        ]
    return [_remove_plugin(claude, dry_run), _remove_marketplace(claude, dry_run)]


def _pi_direct_settings(feature: str) -> tuple[Path, dict, list[str]]:
    """Read direct Pi extension entries that resolve to this feature's source."""
    source, link = PI_LINKS[feature]
    settings_path = link.parent.parent / "settings.json"
    if not settings_path.exists():
        return settings_path, {}, []
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("extensions", []), list):
        raise TypeError("Pi settings must be an object with an extensions list")
    matches = []
    for entry in data.get("extensions", []):
        if not isinstance(entry, str) or entry.startswith(("-", "+", "!", "builtin:")):
            continue
        # pi resolves `..` against the path as written (node's path.resolve), not against a linked `~/.pi`'s target.
        path = Path(os.path.normpath(settings_path.parent / Path(entry).expanduser()))
        if path.resolve() == source.resolve():
            matches.append(entry)
    return settings_path, data, matches


def _write_pi_settings(settings_path: Path, data: dict) -> None:
    if settings_path.is_symlink():
        raise ValueError(f"Pi settings is a symlink: {settings_path}")
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    mode = settings_path.stat().st_mode & 0o777 if settings_path.exists() else 0o600
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=settings_path.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.write("\n")
            tmp.flush()
            os.chmod(tmp_path, mode)
            os.replace(tmp_path, settings_path)
        finally:
            tmp_path.unlink(missing_ok=True)


def _link_pi_extension(feature: str, dry_run: bool = False) -> Step:
    source, link = PI_LINKS[feature]
    label = f"pi extension {link.name}"
    packaged = PACKAGE_ROOT / "pi/compact.ts"
    if not packaged.is_file() or (not dry_run and not source.is_file()):
        return Step(
            label,
            "failed",
            f"missing source: {packaged if not packaged.is_file() else source}",
        )
    if (link.exists() or link.is_symlink()) and not _owned_link(feature):
        target = _link_target(link)
        if target is None or not _is_legacy_pi(target):
            return Step(
                label,
                "failed",
                f"{link} exists and is not a link to {source}; left in place",
            )
    try:
        settings_path, data, direct = _pi_direct_settings(feature)
        if direct and not _owned_link(feature):
            return Step(label, "skipped", "already configured directly in Pi settings")
        if dry_run:
            return Step(
                label, "planned", f"would configure {source} directly in Pi settings"
            )
        if not direct:
            portable = os.path.relpath(source, settings_path.parent)
            data.setdefault("extensions", []).append(portable)
            _write_pi_settings(settings_path, data)
        if _owned_link(feature):
            link.unlink()
    except (OSError, TypeError, ValueError) as err:
        return Step(label, "failed", f"cannot configure Pi extension: {err}")
    return Step(label, "done", "configured directly in Pi settings")


def _remove_pi_extension(
    feature: str, dry_run: bool = False, held_back: bool = False
) -> Step:
    source, link = PI_LINKS[feature]
    label = f"pi extension {link.name}"
    if held_back:
        return Step(label, "skipped", HELD_BACK)
    try:
        settings_path, data, direct = _pi_direct_settings(feature)
        if direct:
            if dry_run:
                return Step(label, "planned", "would remove direct Pi extension entry")
            data["extensions"] = [
                item for item in data["extensions"] if item not in direct
            ]
            _write_pi_settings(settings_path, data)
            if _owned_link(feature):
                link.unlink()
            return Step(label, "done", "removed direct Pi extension entry")
    except (OSError, TypeError, ValueError) as err:
        return Step(label, "failed", f"cannot update Pi settings: {err}")
    if not link.is_symlink() and not link.exists():
        return Step(label, "skipped", f"{link} is not linked")
    if not _owned_link(feature):
        return Step(label, "skipped", f"left {link}: it does not link to {source}")
    if dry_run:
        return Step(label, "planned", f"would remove {link}")
    try:
        link.unlink()
    except OSError as err:
        return Step(label, "failed", f"cannot remove {link}: {err.strerror}")
    return Step(label, "done", f"removed {link}")


def _claude_steps(want: _Wanted, update: bool, dry_run: bool) -> list[Step]:
    """Wire the mod up when claude is selected, and take it back down when it is not."""
    if not want.wants("claude"):
        claude = _claude_cli()
        if claude is None:
            detail = "claude is not selected, and the `claude` CLI is not on PATH"
            return [
                Step("Claude Code mod", "skipped", detail),
                Step("Claude Code marketplace", "skipped", detail),
            ]
        return _unwire_claude(claude, dry_run)
    steps = [_write_marketplace(dry_run)]
    if steps[0].outcome == "failed":
        return [
            *steps,
            Step("Claude Code marketplace", "skipped", HELD_BACK),
            Step("Claude Code mod", "skipped", HELD_BACK),
        ]
    claude = _claude_cli()
    if claude is None:
        steps.append(
            Step("Claude Code marketplace", "failed", "the `claude` CLI is not on PATH")
        )
        steps.append(
            Step("Claude Code mod", "failed", "the `claude` CLI is not on PATH")
        )
        return steps
    steps.append(_add_marketplace(claude, dry_run))
    if steps[-1].outcome == "failed":
        steps.append(
            Step("Claude Code mod", "skipped", "the marketplace step failed first")
        )
    else:
        steps.append(_install_plugin(claude, update, dry_run))
    return steps


def _pi_steps(
    want: _Wanted, dry_run: bool, held_back: bool, reconcile: bool = True
) -> list[Step]:
    """Link the selected pi extensions.

    `reconcile`: an install also takes down the link of every feature it no longer wants. Otherwise (an uninstall)
    the features `want` leaves out are left exactly as they are, link included.
    """
    steps: list[Step] = []
    for (
        feature
    ) in installation.FEATURES:  # compact, memory: one order for every run's output
        link = PI_LINKS[feature][1]
        chosen = want.wants(feature) and want.wants("pi")
        if held_back:
            steps.append(Step(f"pi extension {link.name}", "skipped", HELD_BACK))
            continue
        if reconcile or chosen:
            # An install reconciles both ways; an uninstall only takes down what it was asked for.
            if chosen and reconcile:
                steps.append(_link_pi_extension(feature, dry_run))
            else:
                steps.append(_remove_pi_extension(feature, dry_run, held_back))
        else:
            steps.append(Step(f"pi extension {link.name}", "skipped", "not selected"))
    return steps


def _as_step(step: install_hermes.Step) -> Step:
    """The Hermes module reports its own steps; the installer renders one kind of step."""
    return Step(step.label, step.outcome, step.detail)


def _hermes_steps(
    want: _Wanted,
    dry_run: bool,
    held_back: bool,
    replace_engine: bool = False,
    remove: bool = False,
) -> list[Step]:
    """Link Hermes' plugin and select its engine, or take both back down.

    `remove` is an uninstall: the harnesses `want` leaves out stay exactly as they are, so a `harnesses=[...]` filter
    never takes Hermes out by accident. An install instead reconciles: a deselected hermes is taken back down.
    """
    wanted = want.wants("hermes")
    present = (
        install_hermes.plugin_link().is_symlink()
        or install_hermes.plugin_link().exists()
    )
    source = REPO_ROOT / "hermes"
    if remove:
        if not wanted:
            return [
                Step(label, "skipped", "left in place: not part of this uninstall")
                for label in HERMES_LABELS
            ]
        return [
            _as_step(step)
            for step in install_hermes.uninstall(source=source, dry_run=dry_run)
        ]
    if not wanted and not present:
        return [
            Step(label, "skipped", "hermes is not selected") for label in HERMES_LABELS
        ]
    if held_back:
        return [Step(label, "skipped", HELD_BACK) for label in HERMES_LABELS]
    if not wanted:
        return [
            _as_step(step)
            for step in install_hermes.uninstall(source=source, dry_run=dry_run)
        ]
    steps = [
        _as_step(step)
        for step in install_hermes.install(
            source=source,
            dry_run=dry_run,
            replace_engine=replace_engine,
            validation_source=PACKAGE_ROOT / "hermes",
        )
    ]
    return steps


CODEX_LABEL = "Codex hooks"


def _codex_steps(
    want: _Wanted, dry_run: bool, held_back: bool, remove: bool = False
) -> list[Step]:
    """Add memshort's hooks to Codex's user hooks.json, or take them out; other hooks stay exactly as they are.

    An install also reconciles: a deselected codex loses its hooks. An uninstall leaves codex alone unless asked.
    """
    path = codex.hooks_path()
    wanted = want.wants("codex")
    if remove and not wanted:
        return [
            Step(CODEX_LABEL, "skipped", "left in place: not part of this uninstall")
        ]
    if held_back:
        return [Step(CODEX_LABEL, "skipped", HELD_BACK)]
    try:
        data = codex.read_hooks(path)
        present = codex.without_ours(data) != data
        if remove or not wanted:
            if not present:
                return [Step(CODEX_LABEL, "skipped", f"no memshort hooks in {path}")]
            if dry_run:
                return [
                    Step(
                        CODEX_LABEL,
                        "planned",
                        f"would remove memshort's hooks from {path}",
                    )
                ]
            codex.write_hooks(path, codex.without_ours(data))
            return [Step(CODEX_LABEL, "done", f"removed memshort's hooks from {path}")]
        changed = codex.with_ours(data)
        if changed == data:
            return [Step(CODEX_LABEL, "skipped", f"up to date in {path}")]
        if dry_run:
            return [
                Step(CODEX_LABEL, "planned", f"would add memshort's hooks to {path}")
            ]
        codex.write_hooks(path, changed)
    except (OSError, TypeError, ValueError) as err:
        return [Step(CODEX_LABEL, "failed", f"cannot update {path}: {err}")]
    return [
        Step(
            CODEX_LABEL,
            "updated" if present else "done",
            f"wrote {path}; Codex runs them once you trust them in /hooks",
        )
    ]


def _save_selection(want: installation.Selection, dry_run: bool) -> Step:
    """Write the selection before any harness is touched.

    The mod and the pi extension gate their features on this file, and a host without one falls back to the legacy
    default, both features in both harnesses. Saving first is what keeps a compact-only install from enabling memory
    for the session that lives through the install; the write is idempotent, so a host that already says this only
    reads its own file back.
    """
    label = "install selection"
    path = installation.config_path()
    if dry_run:
        return Step(
            label,
            "planned",
            f"would save {want.describe()} to {path} before touching the harnesses",
        )
    try:
        if installation.exists(path) and installation.load(path) == want:
            return Step(label, "skipped", f"{want.describe()} already saved in {path}")
        written = installation.save(want)
    except installation.SelectionError as err:
        return Step(
            label, "failed", f"{path} cannot be read, so nothing was installed: {err}"
        )
    except OSError as err:
        return Step(
            label,
            "failed",
            f"cannot write {path}: {err.strerror}; nothing was installed",
        )
    return Step(
        label,
        "done",
        f"saved {want.describe()} to {written} before touching the harnesses",
    )


def _unrun_selection(dry_run: bool) -> Step:
    """The selection step of a run that executed nothing: a plan failure, or a failed write (which says why there)."""
    label = "install selection"
    path = installation.config_path()
    if dry_run:
        return Step(
            label,
            "planned",
            f"would save this selection to {path} once the plan is clean",
        )
    return Step(
        label, "skipped", f"not saved: the plan failed first, so {path} is unchanged"
    )


def _result(steps: list[Step]) -> Step:
    """Say plainly how the run ended: there is no rollback, so a partial install is reported as one."""
    failures = [step for step in steps if step.outcome == "failed"]
    if not failures:
        return Step("install result", "done", "complete: every step passed")
    return Step(
        "install result",
        "partial",
        f"{len(failures)} step(s) failed and were not rolled back: what was installed stays, and the selection"
        " saved first is kept, so the harnesses gate what this host asked for rather than the legacy default",
    )


def _legacy_config_path() -> Path:
    override = os.environ.get("MEMSHORT_INSTALL_CONFIG")
    base = Path(
        os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    ).expanduser()
    if override:
        return Path(override).expanduser()
    # The memory product's current selection first, then the pre-rename file it replaced.
    current = base / "memlong/install.json"
    return current if current.is_file() else base / "aimem/install.json"


def _legacy_model() -> str | None:
    """Keep an existing compaction model on the first install, never guess around a broken file."""
    path = _legacy_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as err:
        raise installation.SelectionError(
            f"cannot read legacy selection {path}: {err}; pass --model explicitly"
        ) from err
    if not isinstance(data, dict):
        raise installation.SelectionError(
            f"invalid legacy selection {path}; pass --model explicitly"
        )
    # The memory product names pi models the way version 1 did: through pi, without the prefix.
    model = installation.pi_model(data.get("model", installation.DEFAULT_MODEL))
    try:
        return installation.Selection(model=model).model
    except installation.SelectionError as err:
        raise installation.SelectionError(
            f"invalid model in {path}: {err}; pass --model explicitly"
        ) from err


def _legacy_selected(harness: str) -> bool:
    """Legacy aimem defaults to both features on Claude and pi when its selection is absent."""
    path = _legacy_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return True
    except (OSError, ValueError):
        return True
    if not isinstance(data, dict):
        return True
    features, harnesses = data.get("features"), data.get("harnesses")
    if not isinstance(features, list) or not isinstance(harnesses, list):
        return True
    return "compact" in features and harness in harnesses


def _is_legacy_pi(path: Path) -> bool:
    if path.name != "compact.ts" or path.parent.name != "pi":
        return False
    try:
        project = tomllib.loads(
            (path.parent.parent / "pyproject.toml").read_text(encoding="utf-8")
        )
        return project.get("project", {}).get("name") == "aimem"
    except (OSError, ValueError, AttributeError):
        return False


def _legacy_pi_active() -> bool:
    """Recognize the old aimem compactor without claiming similarly named foreign extensions."""
    source, slot = PI_LINKS["compact"]
    settings = slot.parent.parent / "settings.json"
    entries = []
    if slot.is_symlink():
        entries.append(slot.resolve())
    if settings.exists():
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not isinstance(
                data.get("extensions", []), list
            ):
                return True
            for entry in data.get("extensions", []):
                if not isinstance(entry, str) or entry.startswith(
                    ("-", "+", "!", "builtin:")
                ):
                    continue
                path = os.path.normpath(settings.parent / Path(entry).expanduser())
                entries.append(Path(path).resolve())
        except (OSError, ValueError):
            return True
    for path in entries:
        if path == source.resolve() or not _is_legacy_pi(path):
            continue
        try:
            if "session_before_compact" in path.read_text(encoding="utf-8"):
                return True
        except OSError:
            return True
    return False


def _legacy_catalog(claude: str, *args: str) -> list[dict]:
    result = _run([claude, "plugin", *args, "--json"])
    if result.returncode != 0:
        raise ValueError(f"cannot inspect Claude plugins (exit {result.returncode})")
    try:
        data = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as err:
        raise ValueError("cannot parse Claude plugin list") from err
    if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
        raise ValueError("invalid Claude plugin list")
    return data


def _legacy_conflicts(want: _Wanted) -> Step:
    """Do not install a second compactor next to a still-active pre-split aimem."""
    if want.wants("pi") and _legacy_selected("pi") and _legacy_pi_active():
        return Step(
            "legacy compaction",
            "failed",
            "old aimem compaction is active in pi; upgrade memlong or disable its compact feature first",
        )
    if want.wants("claude") and _legacy_selected("claude"):
        claude = _claude_cli()
        try:
            plugins = _legacy_catalog(claude, "list") if claude else []
        except (OSError, ValueError) as err:
            return Step("legacy compaction", "failed", str(err))
        if claude and any(item.get("id") == "aimem@aimem" for item in plugins):
            try:
                old = next(
                    (
                        item.get("path")
                        for item in _legacy_catalog(claude, "marketplace", "list")
                        if item.get("name") == "aimem"
                    ),
                    None,
                )
            except (OSError, ValueError) as err:
                return Step("legacy compaction", "failed", str(err))
            try:
                register = (
                    (Path(old) / "mod/hooks/register.ts").read_text(encoding="utf-8")
                    if old
                    else None
                )
            except OSError:
                register = None
            if register is None or "registerCompact" in register:
                return Step(
                    "legacy compaction",
                    "failed",
                    "old aimem compaction may be active in Claude Code; upgrade memlong or disable its compact feature first",
                )
    return Step("legacy compaction", "skipped", "no active pre-split compactor found")


def _steps(
    want: _Wanted, update: bool, dry_run: bool, replace_engine: bool = False
) -> list[Step]:
    """Every step in order. With `dry_run` it only reads state, so the plan is the run that would follow it."""
    steps = [_legacy_conflicts(want)]
    if steps[-1].outcome == "failed":
        return steps
    steps.append(_install_resources(dry_run))
    if steps[-1].outcome == "failed":
        return steps
    steps.extend(_claude_steps(want, update, dry_run))
    held_back = any(step.outcome == "failed" for step in steps)
    steps.extend(_pi_steps(want, dry_run, held_back))
    steps.extend(_codex_steps(want, dry_run, held_back))
    steps.extend(_hermes_steps(want, dry_run, held_back, replace_engine))
    return steps


def install(
    update: bool = False,
    features: Iterable[str] | str | None = None,
    harnesses: Iterable[str] | str | None = None,
    model: str | None = None,
    dry_run: bool = False,
    replace_engine: bool = False,
    stage_models: dict[str, str] | None = None,
) -> list[Step]:
    """Wire up the selected features and harnesses, with the selection saved before anything is touched.

    Without explicit arguments the saved selection applies; a host with no config file keeps the old default, both
    features in both harnesses. An explicit argument replaces that part of the selection for this run.

    The order is the contract. The whole run is planned read-only first; a plan that already failed executes nothing
    and saves nothing. Otherwise the desired selection is written **before** the first harness change: the mod and
    the pi extension gate their features on that file, and a host without one falls back to the legacy default, so a
    compact-only install must not run with memory enabled for the session that lives through it. If that write
    fails, nothing else runs.

    There is no rollback. Once the selection is saved, a later failure keeps it — an absent config is never restored
    — because the narrower gates it describes are safer than the legacy default, and it reports a partial install:
    what was wired stays wired, what failed is named, and the next run picks it up. `replace_engine` is the consent to
    switch Hermes' `context.engine` away from another engine than its built-in one; without it the Hermes plugin is
    installed and the setting is left as the user had it.
    """
    try:
        saved = installation.load()
        if model is None and not installation.exists():
            model = _legacy_model()
        want = installation.normalize(
            features=features,
            harnesses=harnesses,
            model=model,
            base=saved,
            stage_models=stage_models,
        )
    except installation.SelectionError as err:
        return [Step("install selection", "failed", str(err))]
    wanted = _Wanted(frozenset(want.features), frozenset(want.harnesses))
    plan = _steps(wanted, update, dry_run=True, replace_engine=replace_engine)
    if dry_run:
        return [*plan, _save_selection(want, dry_run=True)]
    if any(step.outcome == "failed" for step in plan):
        return [*plan, _unrun_selection(dry_run=False)]
    saved_step = _save_selection(want, dry_run=False)
    if saved_step.outcome == "failed":
        return [
            saved_step,
            *[
                Step(
                    step.label,
                    "skipped",
                    "not run: the selection could not be saved first",
                )
                for step in plan
            ],
        ]
    steps = _steps(wanted, update, dry_run=False, replace_engine=replace_engine)
    return [saved_step, *steps, _result(steps)]


def _purge_selection(dry_run: bool = False) -> Step:
    label = "install selection"
    path = installation.config_path()
    if not installation.exists(path):
        return Step(label, "skipped", f"{path} does not exist")
    if dry_run:
        return Step(label, "planned", f"would delete {path}")
    installation.delete(path)
    return Step(label, "done", f"deleted {path}")


def uninstall(
    features: Iterable[str] | str | None = None,
    harnesses: Iterable[str] | str | None = None,
    dry_run: bool = False,
    remove_tool: bool = False,
    purge: bool = False,
) -> list[Step]:
    """Take back what `memshort install` wired up, and nothing else.

    Pi settings entries and legacy links for this checkout, the memshort plugin with its marketplace while the marketplace
    still points at this repo, and Hermes' plugin link with its engine setting (back to the built-in engine). The
    memories, the store, the index, the published copy, the `memshort` uv tool and the saved selection stay: they are
    the user's data and a working command, not install artefacts. `features` and `harnesses` narrow what goes; the
    default removes everything memshort installed in any harness whatever the saved selection says, so an uninstall also
    cleans up after a stale selection. `remove_tool` uninstalls the uv tool as well, `purge` deletes the saved
    selection.

    A harness that is not installed on this host is skipped, not failed: a machine with only pi has no `claude` CLI
    and usually no `~/.hermes`, and `memshort uninstall` there must still succeed. A missing CLI only fails when
    memshort's artifacts are on disk (`_claude_trace`, or Hermes' own plugin link), because then something really was
    left behind that this run cannot remove.
    """
    try:
        chosen_features = set(
            installation.FEATURES
            if features is None
            else installation.features_of(features)
        )
        chosen_harnesses = set(
            installation.HARNESSES
            if harnesses is None
            else installation.harnesses_of(harnesses)
        )
    except installation.SelectionError as err:
        return [Step("uninstall selection", "failed", str(err))]
    want = _Wanted(frozenset(chosen_features), frozenset(chosen_harnesses))
    steps: list[Step] = []
    claude = _claude_cli()
    if not want.wants("claude"):
        steps.extend(
            [
                Step("Claude Code mod", "skipped", "claude is not selected"),
                Step("Claude Code marketplace", "skipped", "claude is not selected"),
            ]
        )
    elif claude is None:
        trace = _claude_trace()
        if trace:
            detail = (
                f"memshort's Claude Code install is still here ({trace[0]}), but the `claude` CLI is not on PATH, so it"
                " cannot be removed; install the CLI or remove that by hand"
            )
            outcome = "failed"
        else:
            detail = "nothing of memshort's is installed in Claude Code, and the `claude` CLI is not on PATH"
            outcome = "skipped"
        steps.extend(
            [
                Step("Claude Code mod", outcome, detail),
                Step("Claude Code marketplace", outcome, detail),
            ]
        )
    else:
        steps.extend(_unwire_claude(claude, dry_run))
    if want.wants("pi"):
        steps.extend(_pi_steps(want, dry_run, held_back=False, reconcile=False))
    else:
        steps.extend(
            Step(f"pi extension {link.name}", "skipped", "pi is not selected")
            for _, link in PI_LINKS.values()
        )
    steps.extend(_codex_steps(want, dry_run, held_back=False, remove=True))
    steps.extend(_hermes_steps(want, dry_run, held_back=False, remove=True))
    if remove_tool:
        steps.append(
            Step(
                "memshort command",
                "failed",
                "run `uv tool uninstall memshort` separately",
            )
        )
    if purge:
        steps.append(_purge_selection(dry_run))
    return steps


def render(steps: list[Step]) -> str:
    width = max(len(s.label) for s in steps)
    lines = [f"{s.label:<{width}}  {s.outcome:<8}  {s.detail}" for s in steps]
    return "\n".join(lines)
