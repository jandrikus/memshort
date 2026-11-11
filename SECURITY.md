# Security policy

## Reporting a vulnerability

Open an issue on the repository.
There is no private reporting channel.

## What memshort keeps

memshort observes older messages of a live session and writes the result back as an observation log.
The adapters are deployed under `~/.local/share/memshort`, and the logs live in the harness's own session files.
Older session text, including tool output, is sent to the provider of the configured model, directly or through pi, and that provider's policy applies to it.
API keys are read from the environment variables an endpoint names, never from the selection file, and `memshort doctor` names the variable, never its value.
A direct call refuses redirects, so a key is never sent to another host.

## Trust boundaries

The adapters run inside pi, Claude Code, Codex and Hermes with the permissions of the user running them.
The Codex hooks keep the observation log of each session in a private file under `$XDG_STATE_HOME/memshort/codex/`.
The selection file is host-local; the deployed adapters are not.
Keep `~/.local/share/memshort` and the harness configuration private to the user.

Conversation text is untrusted, and the observation log is model output derived from it.
A log can carry instructions from an earlier message back into the model's context.
memshort does not sanitize the log for prompt injection.

Compaction is not a secrets-redaction boundary.
Keep credentials out of conversations.
