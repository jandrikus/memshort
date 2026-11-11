"""Observational compaction for live agent sessions."""

import argparse
import json
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="memshort")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("om", help="process a JSON compaction request from stdin")
    hook_parser = sub.add_parser("codex-hook", help="run one Codex lifecycle hook")
    hook_parser.add_argument("event", choices=("pre-compact", "session-start", "stop"))
    install_parser = sub.add_parser(
        "install", help="install the selected harness integrations"
    )
    install_parser.add_argument(
        "--harness", help="comma-separated claude, codex, pi, hermes"
    )
    install_parser.add_argument(
        "--model", help="provider/model for every stage without its own"
    )
    install_parser.add_argument(
        "--observer-model", help="provider/model for the observer only"
    )
    install_parser.add_argument(
        "--reflector-model", help="provider/model for the reflector only"
    )
    install_parser.add_argument("--replace-engine", action="store_true")
    install_parser.add_argument("--update", action="store_true")
    install_parser.add_argument("--dry-run", action="store_true")
    uninstall_parser = sub.add_parser("uninstall", help="remove installed integrations")
    uninstall_parser.add_argument(
        "--harness", help="comma-separated claude, codex, pi, hermes"
    )
    uninstall_parser.add_argument("--dry-run", action="store_true")
    doctor_parser = sub.add_parser(
        "doctor", help="check that each compaction model can be reached"
    )
    doctor_parser.add_argument(
        "--model", help="check this model instead of the selection's"
    )
    doctor_parser.add_argument(
        "--probe",
        action="store_true",
        help="also send one synthetic completion per model",
    )
    endpoint_parser = sub.add_parser(
        "endpoint", help="list, add or remove the HTTP endpoints models can name"
    )
    endpoint_sub = endpoint_parser.add_subparsers(dest="action")
    add_parser = endpoint_sub.add_parser(
        "add", help="define or replace an endpoint on this host"
    )
    add_parser.add_argument("name", help="the name models use: <name>/<model-id>")
    add_parser.add_argument("--api", choices=("openai", "anthropic"), required=True)
    add_parser.add_argument(
        "--base-url", required=True, help="e.g. http://localhost:8080/v1"
    )
    add_parser.add_argument(
        "--api-key-env",
        help="name of the environment variable holding the key, never the key",
    )
    add_parser.add_argument(
        "--max-tokens", type=int, help="output limit sent with each call"
    )
    add_parser.add_argument(
        "--extra-body", help="JSON object merged into each request body"
    )
    remove_parser = endpoint_sub.add_parser("remove", help="remove an endpoint")
    remove_parser.add_argument("name")
    config_parser = sub.add_parser(
        "config", help="show or change this host's models and thresholds"
    )
    config_sub = config_parser.add_subparsers(dest="action")
    set_parser = config_sub.add_parser(
        "set", help="set a model or a threshold in the selection"
    )
    set_parser.add_argument("key")
    set_parser.add_argument("value")
    unset_parser = config_sub.add_parser(
        "unset", help="return a stage model or a threshold to its default"
    )
    unset_parser.add_argument("key")
    args = parser.parse_args()

    if args.command == "codex-hook":
        from memshort import codex

        codex.main(args.event)
    elif args.command == "om":
        from memshort import om

        try:
            print(json.dumps(om.run(json.load(sys.stdin)), ensure_ascii=False))
        except om.ModelError as err:
            print(json.dumps({"error": str(err)}, ensure_ascii=False))
            raise SystemExit(1) from err
    elif args.command == "config":
        from memshort import installation

        try:
            print(_config(installation, args.action, args))
        except installation.SelectionError as err:
            parser.error(str(err))
    elif args.command == "endpoint":
        from memshort import installation

        try:
            print(_endpoint(installation, args))
        except installation.SelectionError as err:
            parser.error(str(err))
    elif args.command == "doctor":
        from memshort import doctor

        report = doctor.check(model=args.model, probe=args.probe)
        print(report.render())
        if not report.ok:
            raise SystemExit(1)
    else:
        from memshort import install, installation

        if args.command == "install":
            if args.harness is None and not installation.exists():
                parser.error(
                    "choose integrations with --harness claude,codex,pi,hermes; this consents to sending session text to the selected model"
                )
            stages = {
                stage: value
                for stage, value in (
                    ("observer", args.observer_model),
                    ("reflector", args.reflector_model),
                )
                if value is not None
            }
            steps = install.install(
                harnesses=args.harness,
                model=args.model,
                dry_run=args.dry_run,
                update=args.update,
                replace_engine=args.replace_engine,
                stage_models=stages or None,
            )
        else:
            steps = install.uninstall(harnesses=args.harness, dry_run=args.dry_run)
        print(install.render(steps))
        if any(step.outcome in ("failed", "partial") for step in steps):
            raise SystemExit(1)


def _endpoint(installation, args) -> str:
    """List the endpoints models can name, or define or remove one in the saved selection."""
    from memshort import backends

    path = installation.config_path()
    if args.action is not None:
        if not installation.exists(path):
            raise installation.SelectionError(
                f"no selection at {path}; run memshort install --harness ... first"
            )
        selection = installation.load(path)
        if args.action == "add":
            try:
                extra = json.loads(args.extra_body) if args.extra_body else {}
                endpoint = backends.endpoint_of(
                    args.name,
                    {
                        "api": args.api,
                        "base_url": args.base_url,
                        **(
                            {"api_key_env": args.api_key_env}
                            if args.api_key_env
                            else {}
                        ),
                        **(
                            {"max_tokens": args.max_tokens}
                            if args.max_tokens is not None
                            else {}
                        ),
                        **({"extra_body": extra} if extra else {}),
                    },
                )
            except (ValueError, backends.BackendError) as err:
                raise installation.SelectionError(str(err)) from None
            changed = installation.with_endpoint(selection, args.name, endpoint)
        else:
            changed = installation.with_endpoint(selection, args.name, None)
        installation.save(changed, path)
    selection = installation.load(path)
    own = selection.endpoint_table()
    lines = [
        f"{backends.PI}: through the pi CLI, every provider and login pi has (pi/<provider>/<model>)"
    ]
    for name, endpoint in sorted({**backends.BUILTINS, **own}.items()):
        origin = "defined here" if name in own else "built in"
        lines.append(f"{name}: {endpoint.describe()} ({origin})")
    if args.action is not None:
        lines.append("restart the harness to apply the change")
    return "\n".join(lines)


def _config(installation, action: str | None, args) -> str:
    """Show every model and threshold with its source, or change one in the saved selection and show the result."""
    path = installation.config_path()
    if action is not None:
        if not installation.exists(path):
            raise installation.SelectionError(
                f"no selection at {path}; run memshort install --harness ... first"
            )
        value = args.value if action == "set" else None
        selection = installation.load(path)
        models = ("model", *(key for key, _, _ in installation.STAGES.values()))
        if args.key in models:
            changed = installation.with_model(selection, args.key, value)
        elif args.key in installation.THRESHOLDS:
            changed = installation.with_threshold(selection, args.key, value)
        else:
            raise installation.SelectionError(
                f"unknown key {args.key!r}; models: {', '.join(models)}; thresholds: {', '.join(installation.THRESHOLDS)}"
            )
        installation.save(changed, path)
    selection = installation.load(path)
    lines = [
        f"{path} (environment wins over file):",
    ]
    shared = os.environ.get(installation.ENV_MODEL, "").strip()
    lines.append(
        f"  model = {selection.model} (file)"
        + (f", overridden by {installation.ENV_MODEL}={shared}" if shared else "")
        + f"  every stage without its own; env {installation.ENV_MODEL}"
    )
    for stage, (key, variable, meaning) in installation.STAGES.items():
        chosen, source = installation.stage_model(stage, selection)
        lines.append(f"  {key} = {chosen} ({source})  {meaning}; env {variable}")
    lines.append("thresholds (tokens):")
    for key, (variable, _, meaning) in installation.THRESHOLDS.items():
        value, source = installation.threshold(key, selection)
        lines.append(f"  {key} = {value} ({source})  {meaning}; env {variable}")
    if action is not None:
        lines.append("restart the harness to apply the change")
    return "\n".join(lines)
