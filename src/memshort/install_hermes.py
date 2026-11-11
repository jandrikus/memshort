"""memshort in Hermes: the context engine, installed as a local plugin of Hermes' own.

Hermes discovers local plugins in `$HERMES_HOME/plugins/<name>/` — a `plugin.yaml` manifest plus an `__init__.py`
whose `register(ctx)` calls `ctx.register_context_engine(...)` — and activates a context engine only when
`context.engine` names it (`compressor`, Hermes' built-in, is the default). So the install is two things:

1. A symlink from `$HERMES_HOME/plugins/memshort` to this repo's `hermes/` plugin, the way the pi extensions are linked,
   so editing the plugin takes effect without a reinstall. Nothing is copied and nothing is overwritten: a directory
   or link the installer did not make is reported and left alone.
2. The `context.engine` setting, changed with `hermes config set context.engine memshort` — the documented interface,
   never by editing `~/.hermes/config.yaml` by hand. An engine the user already configured is not replaced without
   `replace_engine`: the plugin is installed, and the step says which command would switch it. A read of the setting
   that fails (the command errors, or prints something that is not a value) fails the step instead of passing for
   unset: guessing there would overwrite a configured engine without the consent this gate exists for.

Hermes finds the engine through its context-engine discovery (`plugins/context_engine/__init__.py`), which scans
`$HERMES_HOME/plugins/<name>/` for a directory whose `__init__.py` mentions `ContextEngine` and needs no
`plugins.enabled` entry — that opt-in gate (`hermes_cli/plugins_discovery.py:gate_manifest`) covers the *general*
plugin route only. So no `plugins.enabled` / `plugins.entries` key is written, and `context.engine: memshort` is the one
setting that turns the engine on.

Hermes carries the `compact` feature only; the split lives in `memlong.installation`. Uninstalling acts only while the
plugin slot holds this checkout's own link: it restores `context.engine` to Hermes' built-in engine first, and removes
the link only when that restore did not fail — a missing engine would leave Hermes with no compactor at all. A
foreign directory or link in the slot is left exactly as it is.
"""

import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path

# The plugin folder name, the name in its `plugin.yaml` and the `context.engine` value: one name, so discovery, the
# manifest and the setting cannot drift apart.
PLUGIN_NAME = "memshort"
# The plugin this engine replaced, before the rename: install and uninstall migrate it instead of leaving it behind.
LEGACY_PLUGIN_NAME = "aimem"
# Hermes' built-in context engine, and what `context.engine` is set back to on uninstall.
BUILT_IN_ENGINE = "compressor"
MANIFEST = "plugin.yaml"
ENV_HOME = "HERMES_HOME"
_NAME_LINE = re.compile(r"^name:\s*['\"]?([^'\"\s#]+)", re.MULTILINE)


@dataclass
class Step:
    label: str
    outcome: str  # "done", "skipped", "updated", "failed", "planned"
    detail: str = ""


def hermes_home() -> Path:
    """`$HERMES_HOME` when set (Hermes' own override), else `~/.hermes`."""
    override = os.environ.get(ENV_HOME)
    return Path(override).expanduser() if override else Path.home() / ".hermes"


def plugins_dir() -> Path:
    """Hermes' user plugin folder, where a local plugin is a directory or a link."""
    return hermes_home() / "plugins"


def plugin_link() -> Path:
    return plugins_dir() / PLUGIN_NAME


def legacy_plugin_link() -> Path:
    return plugins_dir() / LEGACY_PLUGIN_NAME


def default_source() -> Path:
    """The plugin in this checkout: `<repo>/hermes`."""
    return Path(__file__).resolve().parent / "integrations" / "hermes"


def manifest_name(source: Path) -> str | None:
    """The `name` in the plugin's manifest, or None when it cannot be read. Read without a YAML library."""
    try:
        text = (source / MANIFEST).read_text(encoding="utf-8")
    except OSError:
        return None
    found = _NAME_LINE.search(text)
    return found.group(1) if found else None


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


def installed(source: Path | None = None) -> bool:
    """Whether either plugin slot points at this checkout's `hermes/` plugin."""
    target = Path(source or default_source()).resolve()
    return (
        _link_target(plugin_link()) == target
        or _link_target(legacy_plugin_link()) == target
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def hermes_cli() -> str | None:
    return shutil.which("hermes")


class EngineError(RuntimeError):
    """`context.engine` could not be read: its value is unknown, and it must never be taken for unset."""


def engine(hermes: str) -> str | None:
    """What `context.engine` resolves to; None only when Hermes says the key is unset.

    A failed command, an empty or unparsable answer raises `EngineError` instead of returning None: guessing "unset"
    there would overwrite an engine the user configured without ever asking, and the whole point of the consent gate
    is that it cannot happen by accident.
    """
    ran = _run([hermes, "config", "get", "context.engine", "--json"])
    output = (ran.stdout or "").strip()
    if ran.returncode != 0:
        reason = (ran.stderr or output).strip()[-300:] or f"exit {ran.returncode}"
        raise EngineError(f"`hermes config get context.engine` failed: {reason}")
    if output.startswith("Config key not set"):
        # `hermes config get` prints this line, with exit 0, for a key the user never set.
        return None
    try:
        value = json.loads(output)
    except json.JSONDecodeError as err:
        raise EngineError(
            f"`hermes config get context.engine --json` printed no value: {output[-200:]!r}"
        ) from err
    if value is None:
        return None
    if not isinstance(value, str):
        raise EngineError(f"context.engine is not a name: {value!r}")
    return value


def _set_engine(hermes: str, value: str) -> Step:
    label = "Hermes context engine"
    ran = _run([hermes, "config", "set", "context.engine", value])
    if ran.returncode != 0:
        return Step(label, "failed", (ran.stderr or ran.stdout).strip()[-300:])
    return Step(label, "updated", f"context.engine set to {value}")


def _legacy_plugin(path: Path | None) -> bool:
    """Recognize the pre-split aimem checkout, never a foreign Hermes plugin."""
    if path is None or path.name != "hermes":
        return False
    try:
        data = tomllib.loads(
            (path.parent / "pyproject.toml").read_text(encoding="utf-8")
        )
        return data.get("project", {}).get("name") == "aimem" and manifest_name(
            path
        ) in (PLUGIN_NAME, LEGACY_PLUGIN_NAME)
    except (OSError, ValueError, AttributeError):
        return False


def _install_plugin(
    source: Path, dry_run: bool, validation_source: Path | None = None
) -> Step:
    label = f"hermes plugin {PLUGIN_NAME}"
    checked = validation_source if dry_run and validation_source is not None else source
    if not (checked / MANIFEST).is_file() or not (checked / "__init__.py").is_file():
        return Step(
            label,
            "failed",
            f"missing plugin source: {source} needs {MANIFEST} and __init__.py",
        )
    name = manifest_name(checked)
    if name is not None and name != PLUGIN_NAME:
        return Step(
            label,
            "failed",
            f"{source / MANIFEST} names {name!r}, but the plugin is installed as {PLUGIN_NAME!r}",
        )
    link = plugin_link()
    legacy = legacy_plugin_link()
    if _link_target(link) == source.resolve():
        return Step(label, "skipped", f"{link} already linked")
    if (
        not link.exists()
        and not link.is_symlink()
        and _legacy_plugin(_link_target(legacy))
    ):
        if dry_run:
            return Step(label, "planned", f"would move {legacy} -> {link}")
        try:
            link.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(link)
        except OSError as err:
            return Step(
                label, "failed", f"cannot move {legacy} to {link}: {err.strerror}"
            )
        # The moved link points at the old checkout; the migration below replaces it with this one.
    old_target = _link_target(link)
    migrate = link.is_symlink() and _legacy_plugin(old_target)
    if (link.exists() or link.is_symlink()) and not migrate:
        return Step(
            label,
            "failed",
            f"{link} exists and is not a link to {source}; left in place",
        )
    if dry_run:
        action = "migrate" if migrate else "link"
        return Step(label, "planned", f"would {action} {link} -> {source}")
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if migrate:
            # Replace the old link in one rename, without leaving Hermes without its engine.
            temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
            created = False
            try:
                temporary.symlink_to(source)
                created = True
                os.replace(temporary, link)
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
        else:
            link.symlink_to(source)
    except OSError as err:
        return Step(label, "failed", f"cannot link {link}: {err.strerror}")
    return Step(label, "updated" if migrate else "done", f"{link} -> {source}")


def _remove_plugin(source: Path, dry_run: bool) -> Step:
    label = f"hermes plugin {PLUGIN_NAME}"
    removed = []
    for link in (plugin_link(), legacy_plugin_link()):
        if not link.is_symlink() and not link.exists():
            continue
        if _link_target(link) != source.resolve():
            continue
        if dry_run:
            removed.append(link)
            continue
        try:
            link.unlink()
        except OSError as err:
            return Step(label, "failed", f"cannot remove {link}: {err.strerror}")
        removed.append(link)
    if not removed:
        return Step(label, "skipped", f"{plugin_link()} is not installed")
    if dry_run:
        return Step(
            label, "planned", f"would remove {', '.join(str(p) for p in removed)}"
        )
    return Step(label, "done", f"removed {', '.join(str(p) for p in removed)}")


def _own_link(source: Path) -> bool:
    """Whether the plugin slot holds memshort's own link into this checkout (a foreign file or link is not ours)."""
    return _link_target(plugin_link()) == source.resolve()


def _not_owned_detail(source: Path) -> str:
    link = plugin_link()
    if link.is_symlink():
        return f"left {link}: it does not link to {source}"
    if link.exists():
        return f"left {link}: it was not installed by memshort"
    return f"{link} is not installed"


def _select_engine(dry_run: bool, replace_engine: bool) -> Step:
    label = "Hermes context engine"
    hermes = hermes_cli()
    if hermes is None:
        return Step(
            label,
            "failed",
            "the `hermes` CLI is not on PATH, so `context.engine` cannot be selected",
        )
    try:
        current = engine(hermes)
    except EngineError as err:
        return Step(
            label, "failed", f"cannot read context.engine, left as it is: {err}"
        )
    if current == PLUGIN_NAME:
        return Step(label, "skipped", f"context.engine is already {PLUGIN_NAME}")
    if current == LEGACY_PLUGIN_NAME:
        if dry_run:
            return Step(
                label,
                "planned",
                f"would set context.engine {LEGACY_PLUGIN_NAME} -> {PLUGIN_NAME}",
            )
        step = _set_engine(hermes, PLUGIN_NAME)
        if step.outcome == "updated":
            step.detail = f"context.engine {LEGACY_PLUGIN_NAME} -> {PLUGIN_NAME}"
        return step
    if current not in (None, BUILT_IN_ENGINE) and not replace_engine:
        return Step(
            label,
            "skipped",
            f"context.engine is {current!r}; not replaced without explicit consent"
            f" (`hermes config set context.engine {PLUGIN_NAME}` does it)",
        )
    if dry_run:
        return Step(
            label,
            "planned",
            f"would set context.engine {current or BUILT_IN_ENGINE} -> {PLUGIN_NAME}",
        )
    step = _set_engine(hermes, PLUGIN_NAME)
    if step.outcome == "updated":
        step.detail = f"context.engine {current or BUILT_IN_ENGINE} -> {PLUGIN_NAME}"
    return step


def _restore_engine(dry_run: bool) -> Step:
    label = "Hermes context engine"
    hermes = hermes_cli()
    if hermes is None:
        return Step(
            label,
            "failed",
            "the `hermes` CLI is not on PATH, so `context.engine` cannot be checked",
        )
    try:
        current = engine(hermes)
    except EngineError as err:
        return Step(
            label, "failed", f"cannot read context.engine, left as it is: {err}"
        )
    if current not in (PLUGIN_NAME, LEGACY_PLUGIN_NAME):
        return Step(
            label,
            "skipped",
            f"context.engine is {current or 'unset'}, not {PLUGIN_NAME}",
        )
    if dry_run:
        return Step(
            label, "planned", f"would set context.engine back to {BUILT_IN_ENGINE}"
        )
    step = _set_engine(hermes, BUILT_IN_ENGINE)
    if step.outcome == "updated":
        step.detail = f"context.engine {PLUGIN_NAME} -> {BUILT_IN_ENGINE}"
    return step


def install(
    source: Path | None = None,
    dry_run: bool = False,
    replace_engine: bool = False,
    validation_source: Path | None = None,
) -> list[Step]:
    """Link the plugin in and select it, unless a different engine is configured and consent was not given."""
    source = Path(source) if source is not None else default_source()
    steps = [_install_plugin(source, dry_run, validation_source)]
    if steps[-1].outcome == "failed":
        steps.append(
            Step("Hermes context engine", "skipped", "the plugin step failed first")
        )
        return steps
    steps.append(_select_engine(dry_run, replace_engine))
    return steps


def uninstall(source: Path | None = None, dry_run: bool = False) -> list[Step]:
    """Give `context.engine` back to Hermes' built-in engine first, then remove the plugin link.

    Only memshort's own link is ever acted on: a foreign directory or symlink in the plugin slot is left exactly as it
    is, engine setting included. The restore comes first because removing the plugin while `context.engine` still
    names memshort would leave Hermes with no engine at all, so a failed restore keeps the plugin installed.
    """
    source = Path(source) if source is not None else default_source()
    if not _own_link(source):
        return [
            Step(
                "Hermes context engine",
                "skipped",
                "left as it is: memshort's plugin is not installed here",
            ),
            Step(f"hermes plugin {PLUGIN_NAME}", "skipped", _not_owned_detail(source)),
        ]
    restored = _restore_engine(dry_run)
    if restored.outcome == "failed":
        return [
            restored,
            Step(f"hermes plugin {PLUGIN_NAME}", "skipped", _kept_detail()),
        ]
    return [restored, _remove_plugin(source, dry_run)]


def _kept_detail() -> str:
    return "left in place: the engine could not be restored first, and Hermes must not lose its engine"
