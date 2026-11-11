"""Hermes context-engine plugin: memshort observational compaction (short-term memory only).

Hermes activates this engine only when `context.engine` names `memshort`; installing the plugin and selecting it are
separate steps (`memshort install` asks before replacing another engine). The engine and its state live in this
directory and in `$HERMES_HOME/plugin-data/memshort/`; nothing is written to Hermes' session database.

The plugin imports only the standard library and calls `memshort om` as a subprocess, so no memshort code or compiled
dependency enters Hermes' environment.
"""

from agent.context_engine import ContextEngine

from .engine import (
    PLUGIN_NAME,
    HermesCompactionError,
    ObservationalCompactionEngine,
    is_summary,
    parse_summary,
    render_summary,
)

__all__ = [
    "PLUGIN_NAME",
    "ContextEngine",
    "HermesCompactionError",
    "ObservationalCompactionEngine",
    "is_summary",
    "parse_summary",
    "render_summary",
]


def register(ctx) -> None:
    """Hermes plugin entry point: register one engine; the host clones it per agent."""
    ctx.register_context_engine(ObservationalCompactionEngine())
