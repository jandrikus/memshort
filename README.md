# memshort

Rolling-window compaction for coding agent sessions in pi, Claude Code, Codex and Hermes.
When the context fills up, older messages are replaced by a dated observation log, and the newest messages stay word for word.
There is no long-term memory store. The log lives in the session.

## Why

Harness compaction usually works the same way: when the window is full, a model rewrites the whole past into one summary, at the moment it has the least room to do it well.
Details get lost, and you can't tell which ones.

[Mastra's Observational Memory](https://mastra.ai/research/observational-memory) does it differently.
An observer writes dated notes while the conversation runs, and a reflector condenses those notes when they pile up.
[pi-observational-memory](https://github.com/elpapi42/pi-observational-memory) brought that to pi.
memshort takes the same idea, uses it only for short-term compaction, runs it in several harnesses, and adds a few choices of its own:

- Observation runs in the background every ~10k tokens, so a compaction has little left to do.
- The first user message and every loaded skill are kept verbatim in the log, in conversation order, because a model summary can quietly drop the instructions that matter most.
- If observation fails, compaction fails loudly instead of falling back to the harness's own summary.

memshort covers the short term only.
A long-term memory counterpart, memlong, is planned in a separate repository (not public yet).
The two are meant to work together, but each works on its own.

## How it was made, and why it's public

The ideas are borrowed and adjusted, see above.
The code was written by the pi coding agent under my direction, and I run it on my own pi and Claude Code sessions.
There is no benchmark yet. That it keeps more than normal compaction is my subjective impression from daily use, not a measurement.

I'm publishing it because a rolling context window that keeps as much knowledge as possible deserves more people thinking about it, not because this is the final answer.
If you have a better design, a case where this loses something important, or a way to measure it, open an issue or share your take.
Let's build something better together.

## Install

Install uv and Python 3.13 or later, and the interactive harnesses you want compacted.

```sh
uv tool install .
export DEEPSEEK_API_KEY=...          # or pick another model, see below
memshort install --harness claude,pi
memshort doctor --probe
```

Choose any combination of `claude`, `codex`, `pi` and `hermes`.
`--harness` on the first install is also your consent to send the session text to the configured model provider.
Use `--dry-run` to inspect the changes without applying them.

The installer copies the adapters to `~/.local/share/memshort/integrations` and registers them with the selected harnesses.
The selection is host-local at `$XDG_CONFIG_HOME/memshort/install.json`, or `~/.config/memshort/install.json`.
`MEMSHORT_COMPACT_INSTALL_CONFIG` overrides that path.
A missing or invalid selection never enables an adapter on that host.
Harness settings can be shared between machines, so changes to them may affect other hosts.
Install the tool and select its harnesses on each host that should run compaction.

The plan is checked before changes, then the selection is saved before the adapters are wired.
A later failure is reported as a partial install and is not rolled back.
Rerun the installer to finish it.

Restart the harness after installation.

## Model

`memshort om` makes two kinds of model call: the observer writes observations from the messages a compaction drops, and the reflector condenses the log once it passes `condense_log_above_tokens`.
Each call is one plain completion: memshort's prompt as the system message, the input as one user message, no tools and no history.
Nothing else goes into the request, so you pay for the prompt (under 1,000 tokens) plus the text being observed.

A model is named `<endpoint>/<model-id>`.
The default is `deepseek/deepseek-flash`, called directly with thinking off.

| Endpoint | API | Key from |
|---|---|---|
| `anthropic` | Anthropic Messages, `https://api.anthropic.com` | `MEMSHORT_ANTHROPIC_API_KEY`, else `ANTHROPIC_API_KEY` |
| `openai` | OpenAI chat completions, `https://api.openai.com/v1` | `MEMSHORT_OPENAI_API_KEY`, else `OPENAI_API_KEY` |
| `deepseek` | OpenAI chat completions, `https://api.deepseek.com`, thinking off | `MEMSHORT_DEEPSEEK_API_KEY`, else `DEEPSEEK_API_KEY` |
| `openrouter` | OpenAI chat completions, `https://openrouter.ai/api/v1` | `MEMSHORT_OPENROUTER_API_KEY`, else `OPENROUTER_API_KEY` |
| `ollama` | OpenAI chat completions, `http://localhost:11434/v1` | none needed |
| `pi` | whatever pi uses for `pi/<provider>/<model>` | pi's own login |

The `MEMSHORT_...` variable comes first so compaction can use a key the harness does not see.
Claude Code, for one, bills `ANTHROPIC_API_KEY` instead of your subscription when that variable is set in its environment.

`pi/<provider>/<model>` hands the call to the pi CLI, so it reaches every provider and model in pi's catalog, and every login pi holds, including subscription logins.
memshort turns off every pi resource and replaces pi's system prompt with its own, so the model sees memshort's prompt plus one working-directory line.
Use it when pi already holds the credential you want; otherwise a direct endpoint saves a process start and a dependency.

Any other OpenAI-compatible or Anthropic-compatible server is an endpoint you define on the host: llama.cpp, vLLM, LM Studio, LiteLLM, a company gateway.

```sh
memshort endpoint add lab --api openai --base-url http://localhost:8080/v1
memshort config set reflector_model lab/qwen3-8b
memshort endpoint                      # every endpoint and where its key comes from
memshort doctor --probe
```

`--api-key-env NAME` names the environment variable that holds the key; the key itself never goes in the file.
`--max-tokens` sets the output limit sent with each call.
`--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'` merges fields into every request, for the switch a server needs to turn thinking off.
An endpoint you define with a built-in's name replaces the built-in, so `memshort endpoint add openai --base-url ...` points every `openai/...` model at a gateway.
memshort never reads `OPENAI_BASE_URL` or similar variables, because those often belong to the harness.

| Key in the selection | Environment variable | Used by |
|---|---|---|
| `model` | `MEMSHORT_MODEL` | every stage without its own |
| `observer_model` | `MEMSHORT_OBSERVER_MODEL` | the observer |
| `reflector_model` | `MEMSHORT_REFLECTOR_MODEL` | the reflector |

The most specific setting wins: the stage's variable, the stage's key, `MEMSHORT_MODEL`, then `model`.
Set them with `memshort install --model ... --observer-model ... --reflector-model ...` or `memshort config set reflector_model lab/qwen3-8b`; `memshort config unset reflector_model` returns the stage to `model`.
A malformed selection, an unknown endpoint or a missing key is an error, never a reason to fall back to another provider.

`memshort doctor` checks each model in use: pi and its credentials for a `pi/...` model, the key and the endpoint's model list for a direct one.
It shows which variable a key comes from, never its value.
`--probe` also sends one short synthetic completion per model, which costs a few tokens and may load a local model.
The model's context limit does not resize observation windows.

A selection written before direct endpoints (version 1) named pi models without the prefix.
memshort reads such a file with `pi/` added, so the host keeps the model it had, and writes version 2 the next time it saves.

## Privacy

The observer sends older session messages, including tool output, and the previous observation log to the provider of its model.
Compaction is not a secrets-redaction boundary.
Do not put credentials or material you cannot send to the provider into the conversation.
A local endpoint keeps the text on your machine.

## Behaviour

After a completed turn, compaction starts at about 100,000 raw conversation tokens.
During a turn, the threshold is 200,000.
It retains the newest 15,000 tokens and observes older messages in the background after about 10,000 unobserved tokens.
The observer carries the earlier log forward, and the reflector condenses it above 60,000 estimated tokens.
Harness boundaries and estimates affect the exact cut.
Pi's footer and Claude Code's status line show progress without putting those metrics in the model context.
The raw count is shown against the threshold that applies now: the during-turn one while a turn runs, the after-turn one when idle.
If observation fails, compaction fails with its reason rather than substituting the harness's summary.
Codex works differently, see below.

### Verbatim context

Two kinds of text are kept word for word instead of being turned into observations: the session's first user message, and every skill loaded during the session.
Each one stays in the observation log as a `<memshort-preserved>` block, at the point where it appeared in the conversation.
The observations of what came before it sit above it, and the observations of what came after it sit below it.
The block header gives the text's length, so a skill that quotes `</observation-log>` or any other tag cannot end its block or the log early.

No model writes these blocks.
The observer reads them as clipped, labelled context and is told not to restate them.
The reflector sees only a numbered anchor line in place of each block, and must return every anchor once, in order.
When it does not, memshort condenses each long run of observations between the blocks separately, so a block can never move, merge or disappear.
A kept text is added once: loading the same skill again does not add a second copy.

Kept text is earlier conversation, not a new message.
The summary says so, and says that loaded skills stay in effect and later user messages can change earlier instructions.
Every block is kept until the session ends, because memshort cannot tell when a skill stops mattering.
That grows the context: a session that loads many large skills frees less space at each compaction.
A summary written before this change has no blocks, and the text it already dropped cannot be recovered.
Its log continues as it is, and new kept text is added from the next compaction on.

Skill detection covers only the forms each harness documents for loading a skill:

- Pi: a `/skill:name` expansion (the `<skill name="..." location="...">` block pi puts in the user message), and a successful `read` of a path ending in `SKILL.md`.
- Claude Code: the user message it injects when a skill loads, which starts with `Base directory for this skill:`, and a `Read` of a path ending in `SKILL.md` that returned text.
- Hermes: a successful `skill_view` of a skill's `SKILL.md` (not of its linked files), and a user turn that carries Hermes' slash-skill, skill-bundle or auto-load scaffold (`[IMPORTANT: The user has invoked the "..."` or `[IMPORTANT: The "..." skill is auto-loaded`).

A skill read any other way is observed like any other conversation, not kept: through a shell command such as `cat`, by a subagent or nested tool whose output is not in the transcript, or from a file not named `SKILL.md`.
Only the text of the first user message is kept.
An image in it is not; the observer is told the message carried one.

Set `MEMSHORT_COMPACT=off` to disable the adapters for a process.

## Thresholds

Each threshold has one name in the selection file and a matching environment variable.

| Key in `thresholds` | Environment variable | Default | Meaning |
|---|---|---|---|
| `compact_after_turn_tokens` | `MEMSHORT_COMPACT_AFTER_TURN_TOKENS` | 100000 | Raw conversation that starts a compaction after a completed user turn |
| `compact_during_turn_tokens` | `MEMSHORT_COMPACT_DURING_TURN_TOKENS` | 200000 | Raw conversation that starts a compaction between tool batches of a running turn |
| `keep_recent_tokens` | `MEMSHORT_KEEP_RECENT_TOKENS` | 15000 | Newest raw conversation a compaction keeps unchanged |
| `observe_every_tokens` | `MEMSHORT_OBSERVE_EVERY_TOKENS` | 10000 | Unobserved conversation that starts a background observation |
| `condense_log_above_tokens` | `MEMSHORT_CONDENSE_LOG_ABOVE_TOKENS` | 60000 | Observation log size above which the reflector condenses it |

The values live in the `thresholds` object of the host-local selection, next to the model.
Only the values you set are written there.
The environment variable wins over the file for one process, and the file wins over the default.
An unknown key or a value that is not a positive whole number is an error, and the adapter stays off with the reason.

```sh
memshort config                                          # every threshold, its value and where it comes from
memshort config set compact_during_turn_tokens 250000
memshort config unset compact_during_turn_tokens         # back to the default
```

Restart the harness after a change.

## Hermes

Hermes uses its ContextEngine plugin API and requires consent via `--replace-engine` to replace a different custom engine.
It uses the completed-turn threshold at the next request.
Codex app-server sessions are not supported by the Hermes integration.
Native macOS installation remains unverified.

## Codex

Codex has no hook that can replace its compaction summary, so on Codex memshort adds to it instead of replacing it.
The installer adds three hooks to `$CODEX_HOME/hooks.json` (default `~/.codex/hooks.json`) next to any you already have:

- `PreCompact` reads the session's rollout file and observes everything it has not observed yet, before Codex compacts.
- `SessionStart` with source `compact` runs right after the compaction, also in the middle of a turn, and hands Codex the log as developer context.
- `Stop` observes in the background after a turn once `observe_every_tokens` are unobserved, so a compaction has less left to do.

Codex still writes its own summary and keeps its own recent messages, so the thresholds above do not apply.
Codex compacts at its own limit; set `model_auto_compact_token_limit` in `~/.codex/config.toml` to move it.
Codex drops the injected log at its next compaction and memshort injects the current one, so the context carries one copy.
If observation fails, Codex's own summary stands in and the failure shows as a warning.
memshort keeps the log and how far it read under `$XDG_STATE_HOME/memshort/codex/` (default `~/.local/state/memshort/codex/`), one private file per session.

Codex runs a new hook only after you trust it: open `/hooks` in Codex once after installing.
Subagent sessions are not observed.
The rollout format is not a stable Codex interface; this was tested against Codex 0.167.1.

## Update and uninstall

```sh
uv tool install --reinstall .
memshort install --update
memshort uninstall
uv tool uninstall memshort
```

Uninstall removes owned integrations, including memshort's entries in Codex's `hooks.json`, but leaves the selection, deployed adapter files, and the Hermes and Codex observation data.
Removing the product does not undo compactions already recorded in sessions.

## Development

```sh
uv sync
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
bun test tests/*.test.ts
```

The Hermes runtime lifecycle test is optional and needs `HERMES_AGENT_SRC` pointing at a compatible checkout.
