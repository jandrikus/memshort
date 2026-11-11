"""Host-local selection for memshort's harnesses, model and thresholds.

The file is `$MEMSHORT_COMPACT_INSTALL_CONFIG`, else `$XDG_CONFIG_HOME/memshort/install.json`, else
`~/.config/memshort/install.json`. The installer uses defaults when preparing a first selection, but the
adapters never activate without an explicit host-local file. An invalid file is an error, not a default.

`model` is the model every stage uses unless `observer_model` or `reflector_model` names its own. Each stage's
environment variable wins for one process: `MEMSHORT_OBSERVER_MODEL`, `MEMSHORT_REFLECTOR_MODEL`, then
`MEMSHORT_MODEL` for both. A model is `<endpoint>/<model-id>` (backends.py): `pi/<provider>/<model>` runs through pi,
anything else is an HTTP endpoint, built in or defined in the optional `endpoints` object.

Version 1 named pi models without the `pi/` prefix, since pi made every call. Such a file is read with the prefix
added, so a host keeps the model it had, and is written back as version 2 the next time it is saved.

The optional `thresholds` object holds only the values set on this host; every other threshold is its default.
Each threshold's environment variable wins over the file for one process. The pi extension, the Claude Code mod
and the Hermes engine mirror `THRESHOLDS`; tests/test_thresholds.py keeps the copies equal.
"""

import dataclasses
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memshort import backends

VERSION = 2
# Versions this code reads. Version 1 differs only in how a model is named (see the module docstring).
READS = (1, 2)
FEATURES: tuple[str, ...] = ("compact",)
HARNESSES: tuple[str, ...] = ("claude", "codex", "pi", "hermes")
# Every integration carries compaction; long-term memory is a separate product.
HARNESS_FEATURES: dict[str, tuple[str, ...]] = {
    "claude": FEATURES,
    "pi": FEATURES,
    "codex": ("compact",),
    "hermes": ("compact",),
}
# First-install defaults for selection construction. Adapters still require an explicit host-local file.
DEFAULT_FEATURES = FEATURES
DEFAULT_HARNESSES = ("claude", "pi")
DEFAULT_MODEL = "deepseek/deepseek-flash"
KEYS = (
    "version",
    "features",
    "harnesses",
    "model",
    "observer_model",
    "reflector_model",
    "thresholds",
    "endpoints",
)
# The model calls `memshort om` makes, each with its own optional file key and environment variable.
STAGES: dict[str, tuple[str, str, str]] = {
    "observer": (
        "observer_model",
        "MEMSHORT_OBSERVER_MODEL",
        "writes observations from the transcript a compaction drops",
    ),
    "reflector": (
        "reflector_model",
        "MEMSHORT_REFLECTOR_MODEL",
        "condenses the observation log above condense_log_above_tokens",
    ),
}
ENV_MODEL = "MEMSHORT_MODEL"
# File key: (environment variable, default, what it does). All are token counts.
THRESHOLDS: dict[str, tuple[str, int, str]] = {
    "compact_after_turn_tokens": (
        "MEMSHORT_COMPACT_AFTER_TURN_TOKENS",
        100_000,
        "raw conversation that starts a compaction after a completed user turn",
    ),
    "compact_during_turn_tokens": (
        "MEMSHORT_COMPACT_DURING_TURN_TOKENS",
        200_000,
        "raw conversation that starts a compaction between tool batches of a running turn",
    ),
    "keep_recent_tokens": (
        "MEMSHORT_KEEP_RECENT_TOKENS",
        15_000,
        "newest raw conversation a compaction keeps unchanged",
    ),
    "observe_every_tokens": (
        "MEMSHORT_OBSERVE_EVERY_TOKENS",
        10_000,
        "unobserved conversation older than the kept part that starts a background observation",
    ),
    "condense_log_above_tokens": (
        "MEMSHORT_CONDENSE_LOG_ABOVE_TOKENS",
        60_000,
        "observation log size above which the reflector condenses it",
    ),
}
ENV_PATH = "MEMSHORT_COMPACT_INSTALL_CONFIG"
ENV_XDG = "XDG_CONFIG_HOME"


class SelectionError(ValueError):
    """A selection that cannot be used: an unknown name, an empty list, or a config file that does not parse."""


def features_of(value: Iterable[str] | str) -> tuple[str, ...]:
    """Checked, sorted feature names; a comma or whitespace separated string is accepted."""
    return _names(value, FEATURES, "feature")


def harnesses_of(value: Iterable[str] | str) -> tuple[str, ...]:
    """Checked, sorted harness names; a comma or whitespace separated string is accepted."""
    return _names(value, HARNESSES, "harness")


@dataclass(frozen=True)
class Selection:
    """One install selection: the features, the harnesses and the default model."""

    features: tuple[str, ...] = DEFAULT_FEATURES
    harnesses: tuple[str, ...] = DEFAULT_HARNESSES
    model: str = DEFAULT_MODEL
    # Only the thresholds set on this host, as sorted (key, value) pairs; the rest are defaults.
    thresholds: tuple[tuple[str, int], ...] = ()
    # Only the stage models set on this host, as sorted (stage, model) pairs; the rest use `model`.
    stage_models: tuple[tuple[str, str], ...] = ()
    # The HTTP endpoints defined on this host, as sorted (name, endpoint) pairs; the built-ins need no entry.
    endpoints: tuple[tuple[str, backends.Endpoint], ...] = ()

    def __post_init__(self) -> None:
        features = _names(self.features, FEATURES, "feature")
        harnesses = _names(self.harnesses, HARNESSES, "harness")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "harnesses", harnesses)
        object.__setattr__(self, "endpoints", _endpoints(dict(self.endpoints)))
        object.__setattr__(self, "model", _model(self.model, self.endpoint_table()))
        object.__setattr__(self, "thresholds", _thresholds(dict(self.thresholds)))
        object.__setattr__(
            self,
            "stage_models",
            _stage_models(dict(self.stage_models), self.endpoint_table()),
        )
        _carries(features, harnesses)

    def endpoint_table(self) -> dict[str, backends.Endpoint]:
        return dict(self.endpoints)

    def replace(self, **changes: Any) -> "Selection":
        """This selection with some fields changed, checked again as a whole."""
        return dataclasses.replace(self, **changes)

    def wants(self, name: str) -> bool:
        """Whether the feature or the harness `name` is selected; the two lists never share a name."""
        return name in self.features or name in self.harnesses

    def features_for(self, harness: str) -> tuple[str, ...]:
        """The selected features `harness` can actually run: all of them but on Hermes, compact only."""
        return tuple(
            feature
            for feature in self.features
            if feature in HARNESS_FEATURES.get(harness, ())
        )

    def as_dict(self) -> dict:
        """The JSON shape of the file."""
        data: dict = {
            "version": VERSION,
            "features": list(self.features),
            "harnesses": list(self.harnesses),
            "model": self.model,
        }
        for stage, chosen in self.stage_models:
            data[STAGES[stage][0]] = chosen
        if self.thresholds:
            data["thresholds"] = dict(self.thresholds)
        if self.endpoints:
            data["endpoints"] = {name: e.as_dict() for name, e in self.endpoints}
        return data

    def describe(self) -> str:
        narrowed = [
            f"{harness}: {', '.join(self.features_for(harness))}"
            for harness in self.harnesses
            if len(self.features_for(harness)) < len(self.features)
        ]
        partial = f" ({'; '.join(narrowed)})" if narrowed else ""
        own = "".join(
            f"; {STAGES[stage][0]} {chosen}" for stage, chosen in self.stage_models
        ) + "".join(f"; {key} {value}" for key, value in self.thresholds)
        return f"features {', '.join(self.features)}; harnesses {', '.join(self.harnesses)}{partial}; model {self.model}{own}"


def _carries(features: tuple[str, ...], harnesses: tuple[str, ...]) -> None:
    """Refuse a selection some harness cannot run, or a feature no selected harness carries."""
    for harness in harnesses:
        if set(features) & set(HARNESS_FEATURES[harness]):
            continue
        if harness == "hermes":
            raise SelectionError(
                "Hermes has no long-term memory support: hermes needs the compact feature, or leave hermes out"
            )
        raise SelectionError(
            f"{harness} runs only the {', '.join(HARNESS_FEATURES[harness])} feature"
        )
    for feature in features:
        if any(feature in HARNESS_FEATURES[harness] for harness in harnesses):
            continue
        if feature == "memory" and "hermes" in harnesses:
            raise SelectionError(
                "Hermes has no long-term memory support: the memory feature needs claude or pi, so select one of them"
            )
        raise SelectionError(f"no selected harness runs the {feature} feature")


def _names(
    value: Iterable[str] | str,
    allowed: tuple[str, ...],
    kind: str,
) -> tuple[str, ...]:
    """Sorted, de-duplicated names from `allowed`; a comma or whitespace separated string is accepted too."""
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    try:
        given = list(value)
    except TypeError:
        raise SelectionError(
            f"the {kind} selection must be a string or a list, not {value!r}"
        ) from None
    chosen: list[str] = []
    for name in given:
        if not isinstance(name, str):
            raise SelectionError(f"{kind} names must be strings, not {name!r}")
        key = name.strip().lower()
        if key not in allowed:
            raise SelectionError(
                f"unknown {kind} {name!r}; choose from {', '.join(allowed)}"
            )
        if key in chosen:
            raise SelectionError(f"{kind} {key!r} is selected twice")
        chosen.append(key)
    if not chosen:
        raise SelectionError(f"select at least one {kind}: {', '.join(allowed)}")
    return tuple(sorted(chosen))


def _token_count(value: object, name: str) -> int:
    """A positive whole number of tokens; a string is accepted for the environment and the command line."""
    if isinstance(value, str):
        text = value.strip().replace("_", "")
        value = int(text) if text.isdigit() else value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SelectionError(
            f"{name} must be a positive whole number of tokens, not {value!r}"
        )
    return value


def _thresholds(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, dict):
        raise SelectionError(f"thresholds must be an object, not {value!r}")
    unknown = sorted(set(value) - set(THRESHOLDS))
    if unknown:
        raise SelectionError(
            f"unknown thresholds: {', '.join(map(str, unknown))}; known: {', '.join(THRESHOLDS)}"
        )
    return tuple(sorted((key, _token_count(value[key], key)) for key in value))


def threshold_key(name: str) -> str:
    """A checked threshold key; the error lists the known ones."""
    if name not in THRESHOLDS:
        raise SelectionError(
            f"unknown threshold {name!r}; known: {', '.join(THRESHOLDS)}"
        )
    return name


def threshold(name: str, selection: "Selection | None" = None) -> tuple[int, str]:
    """The value in effect and where it comes from: its environment variable, the file, or the default."""
    variable, default, _ = THRESHOLDS[threshold_key(name)]
    raw = os.environ.get(variable, "").strip()
    if raw:
        return _token_count(raw, variable), variable
    own = dict((load() if selection is None else selection).thresholds)
    if name in own:
        return own[name], "file"
    return default, "default"


def _model(value: object, endpoints: dict[str, backends.Endpoint] | None = None) -> str:
    """A model whose endpoint is known: built in, `pi`, or one of `endpoints`."""
    try:
        return backends.route(value, endpoints).model
    except backends.BackendError as err:
        raise SelectionError(str(err)) from None


def _endpoints(value: dict) -> tuple[tuple[str, backends.Endpoint], ...]:
    out = []
    for name in sorted(value):
        endpoint = value[name]
        if not isinstance(endpoint, backends.Endpoint):
            try:
                endpoint = backends.endpoint_of(name, endpoint)
            except backends.BackendError as err:
                raise SelectionError(str(err)) from None
        else:
            try:
                backends.endpoint_name(name)
            except backends.BackendError as err:
                raise SelectionError(str(err)) from None
        out.append((name, endpoint))
    return tuple(out)


def _stage_models(
    value: dict, endpoints: dict[str, backends.Endpoint] | None = None
) -> tuple[tuple[str, str], ...]:
    unknown = sorted(set(value) - set(STAGES))
    if unknown:
        raise SelectionError(
            f"unknown model stages: {', '.join(map(str, unknown))}; known: {', '.join(STAGES)}"
        )
    out = []
    for stage in sorted(value):
        try:
            out.append((stage, _model(value[stage], endpoints)))
        except SelectionError as err:
            raise SelectionError(f"{STAGES[stage][0]}: {err}") from err
    return tuple(out)


def stage_key(name: str) -> str:
    """The stage a file key such as `observer_model` names; the error lists the known keys."""
    for stage, (key, _, _) in STAGES.items():
        if name == key:
            return stage
    raise SelectionError(
        f"unknown model key {name!r}; known: model, {', '.join(key for key, _, _ in STAGES.values())}"
    )


def stage_model(stage: str, selection: "Selection | None" = None) -> tuple[str, str]:
    """The model a stage uses and where it comes from. Never falls back on error.

    The stage's environment variable, the stage's key in the file, `$MEMSHORT_MODEL`, the file's `model`, the
    default: the most specific setting wins, and an environment variable wins over the file at the same level.
    The source is the variable's name, `file` for the stage key, or `model` when the stage uses the shared model.
    """
    if stage not in STAGES:
        raise SelectionError(f"unknown model stage: {stage!r}")
    _, variable, _ = STAGES[stage]
    chosen = load() if selection is None else selection
    for name in (variable, None, ENV_MODEL):
        if name is None:
            own = dict(chosen.stage_models)
            if stage in own:
                return own[stage], "file"
            continue
        value = os.environ.get(name, "").strip()
        if value:
            try:
                return _model(value, chosen.endpoint_table()), name
            except SelectionError as err:
                raise SelectionError(f"{name}: {err}") from err
    return chosen.model, "model"


def config_path() -> Path:
    """Where the selection is saved: the test override, else the XDG config home, else `~/.config`."""
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override).expanduser()
    base = os.environ.get(ENV_XDG)
    root = Path(base).expanduser() if base else Path.home() / ".config"
    return root / "memshort" / "install.json"


def exists(path: Path | None = None) -> bool:
    return (config_path() if path is None else path).is_file()


def load(path: Path | None = None) -> Selection:
    """The saved selection; an absent file is the defaults, anything malformed is an error that names the file."""
    path = config_path() if path is None else path
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Selection()
    except OSError as err:
        raise SelectionError(f"cannot read {path}: {err.strerror}") from err
    try:
        data = json.loads(text)
    except json.JSONDecodeError as err:
        raise SelectionError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(data, dict):
        raise SelectionError(f"{path} must hold a JSON object")
    data = dict[str, Any](data)
    unknown = sorted(set(data) - set(KEYS))
    if unknown:
        raise SelectionError(
            f"{path} has unknown keys: {', '.join(unknown)}; known: {', '.join(KEYS)}"
        )
    version = data.get("version", VERSION)
    if version not in READS:
        raise SelectionError(
            f"{path} is version {version!r}, this memshort reads versions {', '.join(map(str, READS))}"
        )
    if version == 1:
        if "endpoints" in data:
            raise SelectionError(f"{path}: endpoints need version {VERSION}")
        data.setdefault("model", DEFAULT_MODEL)
        for key in ("model", *(k for k, _, _ in STAGES.values())):
            if key in data:
                data[key] = pi_model(data[key])
    missing = [key for key in ("features", "harnesses") if key not in data]
    if missing:
        raise SelectionError(f"{path} is missing {', '.join(missing)}")
    for key in ("features", "harnesses"):
        if not isinstance(data[key], list):
            raise SelectionError(f"{path}: {key} must be a list, not {data[key]!r}")
    try:
        return Selection(
            data["features"],
            data["harnesses"],
            data.get("model", DEFAULT_MODEL),
            _thresholds(data.get("thresholds", {})),
            tuple(
                (stage, data[key])
                for stage, (key, _, _) in STAGES.items()
                if key in data
            ),
            _endpoint_object(data.get("endpoints", {})),
        )
    except SelectionError as err:
        raise SelectionError(f"{path}: {err}") from err


def _endpoint_object(value: object) -> Any:
    if not isinstance(value, dict):
        raise SelectionError(f"endpoints must be an object, not {value!r}")
    return tuple(value.items())


def pi_model(value: Any) -> Any:
    """A version 1 model, `provider/model` as pi names it, as the `pi/` model it was."""
    return f"{backends.PI}/{value.strip()}" if isinstance(value, str) else value


def save(selection: Selection, path: Path | None = None) -> Path:
    """Write the selection atomically: a killed install leaves either the old file or the new one, never half."""
    path = config_path() if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(selection.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def delete(path: Path | None = None) -> bool:
    """Forget the selection; returns whether there was one. A host without a file just gets the defaults again."""
    path = config_path() if path is None else path
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def normalize(
    features: Iterable[str] | str | None = None,
    harnesses: Iterable[str] | str | None = None,
    model: str | None = None,
    base: Selection | None = None,
    stage_models: dict[str, str] | None = None,
) -> Selection:
    """`base` with the explicit parts replaced; an argument left None keeps what was saved.

    `stage_models` sets the named stages and keeps the others as saved."""
    base = Selection() if base is None else base
    return base.replace(
        features=base.features if features is None else features_of(features),
        harnesses=base.harnesses if harnesses is None else harnesses_of(harnesses),
        model=base.model if model is None else model,
        stage_models=tuple({**dict(base.stage_models), **(stage_models or {})}.items()),
    )


def with_threshold(base: Selection, name: str, value: int | str | None) -> Selection:
    """`base` with one threshold set, or back to its default when `value` is None."""
    own = dict(base.thresholds)
    if value is None:
        own.pop(threshold_key(name), None)
    else:
        own[threshold_key(name)] = _token_count(value, name)
    return base.replace(thresholds=tuple(own.items()))


def with_model(base: Selection, name: str, value: str | None) -> Selection:
    """`base` with `model` or one stage model set; a stage model goes back to `model` when `value` is None."""
    if name == "model":
        if value is None:
            raise SelectionError(
                "model cannot be unset; set it to another provider/model"
            )
        return base.replace(model=value)
    stage = stage_key(name)
    own = dict(base.stage_models)
    if value is None:
        own.pop(stage, None)
    else:
        own[stage] = value
    return base.replace(stage_models=tuple(own.items()))


def with_endpoint(
    base: Selection, name: str, endpoint: backends.Endpoint | None
) -> Selection:
    """`base` with one endpoint defined or replaced, or removed when `endpoint` is None.

    A removal that leaves a model without its endpoint is an error naming that model."""
    own = dict(base.endpoints)
    if endpoint is None:
        if name not in own:
            raise SelectionError(f"no endpoint {name!r} is defined on this host")
        own.pop(name)
    else:
        own[name] = endpoint
    return base.replace(endpoints=tuple(own.items()))
