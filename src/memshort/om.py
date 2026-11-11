"""Short-term observational memory: the compaction of a live session, shared by the pi extension and the mod.

A harness hands over the session's observation log, the current task and a slice of transcript it is about to
drop; the observer (deepseek-flash by default, thinking off) appends what the slice holds, and the reflector condenses the
log once it grows past `condense_log_above_tokens` (`reflect_tokens()`). The harness puts `summary` in place of the dropped messages.
When to observe and what to keep raw is the harness's call (pi/compact.ts, mod/hooks/compact.tsx).

The transcript is either one string or an ordered list of parts: `{"text": ...}` for conversation the observer
turns into observations, and `{"preserved": {"kind": "user"|"skill", "name": ..., "content": ...}}` for text the
harness wants kept verbatim (the first user message, a loaded skill). A preserved item goes into the log as a
length-prefixed `<memshort-preserved>` block at the point it appeared, between the observations of the text before
and after it. No model writes those blocks: the observer sees them as context, the reflector only as anchors.
"""

import json
import re
from datetime import datetime
from pathlib import Path

from memshort import backends, config, installation

PROMPTS = Path(__file__).resolve().parent / "prompts"
DEFAULT_MODEL = installation.DEFAULT_MODEL
# Kept for callers that imported the old constant. `complete` resolves the model through `model()`, so this is only
# the default, not the value a run uses.
MODEL = DEFAULT_MODEL
# The shared per-run override, below each stage's own (`installation.STAGES`) and above the install selection.
ENV_MODEL = installation.ENV_MODEL
# Condense the accumulated log above this estimated size, independently of the raw-conversation trigger. This is the
# default; `reflect_tokens()` is the value a run uses.
REFLECT_TOKENS = installation.THRESHOLDS["condense_log_above_tokens"][1]
TIMEOUT_S = 300
# One completion is tried twice, and a run can do both a completion for the observer and one for the reflector:
# 2 attempts * 2 completions * 300 s. A harness that starts `memshort om` has to allow for this; the adapters' own
# limits are theirs to change (see the report). A transcript with preserved parts takes one observer completion per
# text run between them, so such a run can take longer.
MAX_RUN_S = 2 * 2 * TIMEOUT_S

# A preserved block in the log: the open tag and a one-line JSON header (`kind`, optional `name`, `length` in code
# points), the content, then the close tag on its own line. The length, not the close tag, ends the content, so a
# skill that quotes any tag cannot end its block or the log early.
BLOCK_OPEN = "<memshort-preserved>"
BLOCK_CLOSE = "</memshort-preserved>"
KINDS = ("user", "skill")
# The observer sees at most this much of each preserved block; it is context, not material to observe.
VIEW_CHARS = 2000
# In the per-segment fallback, the reflector only rewrites runs of observations at least this long.
SEGMENT_REFLECT_CHARS = 4000
ANCHOR_RE = re.compile(r"^⟦preserved (\d+)⟧[ \t]*$", re.MULTILINE)

# Greedy: the block runs to the reply's last closing tag, so a tag quoted inside it cannot end it early.
OBSERVATIONS_RE = re.compile(r"<observations>\s*(.*)\s*</observations>", re.DOTALL)
TASK_RE = re.compile(r"<current-task>\s*(.*?)\s*</current-task>", re.DOTALL)
BULLET_RE = re.compile(r"^[*-]\s+(🔴|🟡|🟢)\s*(?:\(\d{1,2}:\d{2}\)\s*)?(.*)$")
DATE_RE = re.compile(r"^Date: (.+)$", re.MULTILINE)
# The memory block memlong (and, before the rename, aimem) splices into a user turn is not conversation.
CONTEXT_RE = re.compile(
    r"<(?:memlong|aimem)-context>.*?</(?:memlong|aimem)-context>\s*", re.DOTALL
)
# The tags the summary is built and parsed with. A transcript can carry them (a file or page that contains them),
# and the model may echo them. Their brackets are escaped going in and coming out, so none can form in the log;
# escaping in one pass cannot reassemble a tag the way removing one can.
TAG_RE = re.compile(
    r"</?((?:memshort|aimem)-observations|memshort-preserved|(?:memlong|aimem)-context|observation-log|current-task|observations|focus|conversation)>"
)


def neutral(text: str) -> str:
    """Escape the summary's tags and the reflector's anchor brackets, so model text can form neither."""
    text = TAG_RE.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), text)
    return text.replace("⟦", "[").replace("⟧", "]")


class ModelError(Exception):
    pass


def model(stage: str = "observer") -> str:
    """The model `stage` (observer or reflector) uses; `installation.stage_model` gives the order.

    A missing selection file is the default, as `installation.load` says. A file that does not parse, names a bad
    model or cannot be read is a `ModelError` that names the cause, and so is a bad environment override.
    """
    try:
        return installation.stage_model(stage)[0]
    except installation.SelectionError as err:
        raise ModelError(
            f"cannot read the {stage} model from the install selection: {err}"
        ) from err


def reflect_tokens() -> int:
    """The log size that starts the reflector: `$MEMSHORT_CONDENSE_LOG_ABOVE_TOKENS`, the selection, the default."""
    try:
        return installation.threshold("condense_log_above_tokens")[0]
    except installation.SelectionError as err:
        raise ModelError(f"cannot read the reflector threshold: {err}") from err


def tokens(text: str) -> int:
    return int(len(text) / config.CHARS_PER_TOKEN)


Segment = str | dict


def split_log(log: str) -> list[Segment]:
    """The log as runs of observation text (str) and preserved blocks (dict with kind, name, content), in order.

    Observation text is always escaped by `neutral`, so a block opens only where the open tag starts a line.
    """
    segments: list[Segment] = []
    pos = text_start = 0
    while True:
        at = log.find(BLOCK_OPEN, pos)
        if at < 0:
            break
        if at > 0 and log[at - 1] != "\n":
            pos = at + 1
            continue
        try:
            end = log.index("\n", at)
            header = json.loads(log[at + len(BLOCK_OPEN) : end])
            length = header.pop("length")
            if (
                header.get("kind") not in KINDS
                or not isinstance(length, int)
                or isinstance(length, bool)
                or length < 0
                or not isinstance(header.get("name", ""), str)
                or set(header) - {"kind", "name"}
            ):
                raise ValueError("bad header")
        except (ValueError, KeyError, AttributeError) as err:
            raise ModelError("invalid preserved block in the observation log") from err
        body = end + 1
        close = body + length
        if log[close : close + 1 + len(BLOCK_CLOSE)] != "\n" + BLOCK_CLOSE:
            raise ModelError("invalid preserved block length in the observation log")
        if log[text_start:at].strip():
            segments.append(log[text_start:at].strip("\n"))
        segments.append({**header, "content": log[body:close]})
        pos = text_start = close + 1 + len(BLOCK_CLOSE)
    if log[text_start:].strip():
        segments.append(log[text_start:].strip("\n"))
    return segments


def block(item: dict) -> str:
    header = {"kind": item["kind"]}
    if item.get("name"):
        header["name"] = item["name"]
    header["length"] = len(item["content"])
    return f"{BLOCK_OPEN}{json.dumps(header, ensure_ascii=False)}\n{item['content']}\n{BLOCK_CLOSE}"


def join_log(segments: list[Segment]) -> str:
    return "\n".join(
        s.strip("\n") if isinstance(s, str) else block(s)
        for s in segments
        if not isinstance(s, str) or s.strip()
    )


def observations_of(log: str) -> str:
    """The model-written part of the log, without the preserved blocks."""
    return "\n".join(s for s in split_log(log) if isinstance(s, str))


def label(item: dict) -> str:
    if item["kind"] == "user":
        return "the first user message"
    return f"skill {item['name']}" if item.get("name") else "a loaded skill"


def observer_view(log: str) -> str:
    """The log as the observer reads it: observations as they are, each preserved block labelled and clipped."""
    out: list[str] = []
    n = 0
    for segment in split_log(log):
        if isinstance(segment, str):
            out.append(neutral(segment))
            continue
        n += 1
        content = segment["content"]
        if len(content) > VIEW_CHARS:
            content = f"{content[:VIEW_CHARS]} [{len(content) - VIEW_CHARS} more chars]"
        out.append(
            f"⟦preserved {n}: {label(segment)}, kept verbatim⟧\n{neutral(content)}\n⟦end of preserved {n}⟧"
        )
    return "\n".join(out)


def parts_of(transcript: object) -> list[Segment]:
    """A request's transcript as text runs and preserved items; a string is one text run."""
    if transcript is None:
        return []
    if isinstance(transcript, str):
        return [transcript]
    if not isinstance(transcript, list):
        raise ModelError("invalid request: transcript must be a string or a list")
    parts: list[Segment] = []
    for part in transcript:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
            continue
        item = part.get("preserved") if isinstance(part, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("kind") not in KINDS
            or not isinstance(item.get("content"), str)
            or not isinstance(item.get("name", ""), str)
        ):
            raise ModelError(
                "invalid request: a transcript part is {text} or {preserved: {kind, name, content}}"
            )
        kept = {"kind": item["kind"], "content": item["content"]}
        if item.get("name"):
            kept["name"] = item["name"]
        parts.append(kept)
    return parts


def route(stage: str) -> backends.Route:
    """Where `stage`'s calls go, resolved from the selection; a bad model or endpoint is a `ModelError`."""
    try:
        selection = installation.load()
        return backends.route(
            installation.stage_model(stage, selection)[0], selection.endpoint_table()
        )
    except (installation.SelectionError, backends.BackendError) as err:
        raise ModelError(
            f"cannot read the {stage} model from the install selection: {err}"
        ) from err


def complete(role: str, prompt: str) -> tuple[str, dict]:
    """One completion with no tools: the reply's text and its usage in pi's shape. Retried once, as a provider
    drops a stream now and then, and an observer reply without its block is asked for again."""
    prompt_file = PROMPTS / f"om-{role}.md"
    try:
        system_prompt = prompt_file.read_text(encoding="utf-8")
    except OSError as err:
        raise ModelError(
            f"cannot read the {role} prompt at {prompt_file}: {err.strerror or err}"
        ) from err
    chosen = route(role)
    error = ""
    total_usage: dict = {}
    for _ in range(2):
        try:
            text, usage = backends.complete(chosen, system_prompt, prompt, TIMEOUT_S)
        except backends.BackendError as err:
            error = str(err)
            if "not on PATH" in error or error.startswith("no API key"):
                break  # the same call fails the same way again
            continue
        total_usage = add_usage(total_usage, usage)
        if role == "observer":
            try:
                observer_block(text)
            except ModelError as err:
                error = str(err)
                continue
        return text, total_usage
    raise ModelError(f"{chosen.model}: {error}")


COST_KEYS = ("input", "output", "cacheRead", "cacheWrite", "total")


def add_usage(total: dict, usage: dict) -> dict:
    """Sum pi usage records (`input`, `output`, `cacheRead`, `cacheWrite`, `totalTokens`, `cost`).

    Every cost key is always present: pi reads `usage.cost.total` without a guard, so a record with an empty
    cost object takes the harness down (pi 1.4.0, `addCompactionCostNotice` and the footer's session stats).
    """
    out = dict(total)
    for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens"):
        out[key] = out.get(key, 0) + (usage.get(key) or 0)
    cost = dict(out.get("cost") or {})
    for key, value in (usage.get("cost") or {}).items():
        cost[key] = cost.get(key, 0) + (value or 0)
    out["cost"] = {key: cost.get(key, 0) for key in COST_KEYS}
    return out


def stamp(observations: str, log: str, now: datetime) -> str:
    """Put the observer's lines under today's date header and give each top-level line the time it was observed."""
    lines = []
    for line in observations.splitlines():
        if not line.strip():
            continue
        if line.startswith("Date:"):  # the harness owns dates
            continue
        m = BULLET_RE.match(line.strip()) if not line[0].isspace() else None
        lines.append(
            f"* {m.group(1)} ({now:%H:%M}) {m.group(2).strip()}" if m else line.rstrip()
        )
    if not lines:
        return ""
    dates = DATE_RE.findall(log)
    today = f"{now:%A, %b} {now.day}, {now.year}"
    head = [] if dates and dates[-1].strip() == today else [f"Date: {today}"]
    return "\n".join(head + lines)


def observer_block(reply: str) -> tuple[str, int]:
    """Find the block, or recover its missing close only when a complete final task marks the boundary."""
    found = OBSERVATIONS_RE.search(reply)
    if found and "<observations>" not in reply[found.end() :]:
        return found.group(1), found.end()

    opening = reply.find("<observations>")
    if opening < 0:
        reason = "missing <observations>"
    else:
        reason = "missing </observations> or ambiguous observation block"
        task = re.search(
            r"(?m)^<current-task>\s*(.*?)\s*</current-task>\s*\Z", reply, re.DOTALL
        )
        start = opening + len("<observations>")
        if (
            task
            and start <= task.start()
            and reply.count("<observations>") == 1
            and "</observations>" not in reply
        ):
            return reply[start : task.start()], task.start()
    # The reply may contain private conversation content. Show only recognized structural tags from its tail.
    tail_tags = [match.group() for match in TAG_RE.finditer(reply[-300:])]
    raise ModelError(f"observer reply {reason}; tail tags: {tail_tags}")


def observe_text(
    log: str, task: str, transcript: str, focus: str | None, now: datetime
) -> tuple[str, str, dict]:
    """Observe one run of conversation: the new, stamped observations, the current task and the usage."""
    parts = [
        f"Now: {now:%Y-%m-%d %H:%M (%A)}",
        f"<observation-log>\n{observer_view(log).strip() or '(empty)'}\n</observation-log>",
        f"<current-task>\n{neutral(task).strip() or '(none)'}\n</current-task>",
    ]
    if focus:
        parts.append(f"<focus>\n{neutral(focus).strip()}\n</focus>")
    parts.append(
        f"<conversation>\n{neutral(CONTEXT_RE.sub('', transcript)).strip()}\n</conversation>"
    )
    reply, usage = complete("observer", "\n\n".join(parts))
    observations, end = observer_block(reply)
    new = stamp(neutral(observations), observations_of(log), now)
    # The last task block after the observations: one the observations quote cannot stand in for it.
    tasks = TASK_RE.findall(reply, end)
    if tasks:
        text = neutral(tasks[-1]).strip()
        task = "" if text.lower() in ("none", "(none)") else text
    return new, task, usage


def keep_item(item: dict) -> dict:
    """A preserved item as it is stored. A user message loses the memory block memlong spliced into it."""
    if item["kind"] == "user":
        item = {**item, "content": CONTEXT_RE.sub("", item["content"])}
    return item


def observe(
    log: str, task: str, transcript: object, focus: str | None, now: datetime
) -> tuple[str, str, dict | None]:
    """Append the slice to the log in order: observations for each text run, preserved items verbatim between them.

    An item already preserved in the log (same kind and content) is not added again. The usage is None when no
    text run needed a model call.
    """
    segments = split_log(log)
    seen = {(s["kind"], s["content"]) for s in segments if isinstance(s, dict)}
    usage: dict | None = None
    for part in parts_of(transcript):
        if isinstance(part, str):
            if not neutral(CONTEXT_RE.sub("", part)).strip():
                continue
            new, task, used = observe_text(join_log(segments), task, part, focus, now)
            usage = add_usage(usage or {}, used)
            if new:
                segments.append(new)
            continue
        item = keep_item(part)
        if not item["content"].strip() or (item["kind"], item["content"]) in seen:
            continue
        seen.add((item["kind"], item["content"]))
        segments.append(item)
    return join_log(segments), task, usage


def _condensed(reply: str) -> str | None:
    found = OBSERVATIONS_RE.search(reply)
    return found.group(1) if found and found.group(1).strip() else None


def reflect_text(text: str) -> tuple[str, dict]:
    """Condense one run of observations. A reply that is empty or no shorter keeps the old text."""
    reply, usage = complete(
        "reflector", f"<observation-log>\n{text.strip()}\n</observation-log>"
    )
    new = _condensed(reply)
    if new is None or len(new) >= len(text):
        return text, usage
    return neutral(new).strip(), usage


def reflect(log: str) -> tuple[str, dict]:
    """Condense the observations and leave every preserved block as it is, where it is.

    The reflector sees each block as an anchor line and has to return every anchor once, in order. When it does not,
    each long run of observations between the blocks is condensed on its own instead, so no block can move.
    """
    segments = split_log(log)
    blocks = [s for s in segments if isinstance(s, dict)]
    if not blocks:
        return reflect_text(log)
    anchored: list[str] = []
    n = 0
    for segment in segments:
        if isinstance(segment, str):
            anchored.append(segment)
        else:
            n += 1
            anchored.append(f"⟦preserved {n}⟧")
    old = sum(len(s) for s in segments if isinstance(s, str))
    reply, usage = complete(
        "reflector", f"<observation-log>\n{chr(10).join(anchored)}\n</observation-log>"
    )
    new = _condensed(reply)
    if new is not None:
        anchors = list(ANCHOR_RE.finditer(new))
        if [int(a.group(1)) for a in anchors] == list(range(1, len(blocks) + 1)):
            pieces = [neutral(new[: anchors[0].start()]).strip()]
            for a, b in zip(anchors, anchors[1:] + [None], strict=True):
                pieces.append(neutral(new[a.end() : b.start() if b else None]).strip())
            if sum(len(p) for p in pieces) < old:
                out: list[Segment] = [pieces[0]]
                for item, piece in zip(blocks, pieces[1:], strict=True):
                    out += [item, piece]
                return join_log(out), usage
    out = []
    for segment in segments:
        if isinstance(segment, str) and len(segment) >= SEGMENT_REFLECT_CHARS:
            segment, used = reflect_text(segment)
            usage = add_usage(usage, used)
        out.append(segment)
    return join_log(out), usage


PRESERVED_NOTE = (
    "Each <memshort-preserved> block in the log is verbatim text from the conversation, at the point where it "
    "appeared: the first user message or a skill that was loaded. It is earlier conversation, not a new message. "
    "Loaded skills stay in effect; later user messages can change earlier instructions. The block's header gives "
    "its length, so a tag inside its text does not end it.\n"
)


def render(log: str, task: str) -> str:
    """The message that stands in for the dropped part of the conversation; `parse` reads it back."""
    note = PRESERVED_NOTE if any(isinstance(s, dict) for s in split_log(log)) else ""
    return (
        "<memshort-observations>\n"
        "The earlier part of this conversation was replaced by these observations, oldest first. "
        "They are your memory of it; the messages after this one continue the conversation.\n"
        "🔴 the user's requests, decisions and constraints; 🟡 work done and findings; 🟢 minor context.\n"
        f"{note}\n"
        f"<observation-log>\n{log.strip() or '(empty)'}\n</observation-log>\n\n"
        f"<current-task>\n{task.strip() or '(none)'}\n</current-task>\n"
        "</memshort-observations>"
    )


LOG_OPEN = "<observation-log>\n"
LOG_END = "</observation-log>"


def log_span(summary: str) -> tuple[int, int] | None:
    """Where the log's text starts and ends in a rendered summary. It is read line by line, and each preserved
    block is stepped over by its length, so a closing tag quoted inside one does not end the log."""
    start = summary.find(LOG_OPEN)
    if start < 0:
        return None
    start += len(LOG_OPEN)
    pos = start
    while pos < len(summary):
        if summary.startswith(BLOCK_OPEN, pos):
            try:
                end = summary.index("\n", pos)
                length = json.loads(summary[pos + len(BLOCK_OPEN) : end])["length"]
            except (ValueError, KeyError, TypeError):
                return None
            if not isinstance(length, int) or isinstance(length, bool) or length < 0:
                return None
            pos = end + 1 + length + 1
            if not summary.startswith(BLOCK_CLOSE, pos):
                return None
            pos += len(BLOCK_CLOSE)
        else:
            line_end = summary.find("\n", pos)
            line_end = len(summary) if line_end < 0 else line_end
            if summary[pos:line_end] == LOG_END:
                return start, max(start, pos - 1)
            pos = line_end
        if summary.startswith("\n", pos):
            pos += 1
        else:
            return None
    return None


def parse(summary: str) -> tuple[str, str] | None:
    """The log and task of a rendered summary, or None when the text is not one."""
    if not any(
        tag in summary for tag in ("<memshort-observations>", "<aimem-observations>")
    ):
        return None
    span = log_span(summary)
    log = summary[span[0] : span[1]] if span else ""
    found = TASK_RE.search(summary, span[1] if span else 0)
    task = found.group(1).strip() if found else ""
    return ("" if log.strip() == "(empty)" else log.strip("\n")), (
        "" if task == "(none)" else task
    )


def run(request: dict, now: datetime | None = None) -> dict:
    """`memshort om`: JSON in (`log`, `task`, `transcript`, `focus`), JSON out (`log`, `task`, `summary`, `usage`,
    `reflected`). The transcript is a string or a list of parts (see the module docstring). With no transcript it
    only renders, and reflects if the log is over the limit. `usage` is null when no model call was made."""
    now = now or datetime.now().astimezone()
    log, task = request.get("log") or "", request.get("task") or ""
    split_log(log)  # a log whose blocks do not parse is an error before any model call
    usage = None
    if request.get("transcript"):
        log, task, used = observe(
            log, task, request["transcript"], request.get("focus"), now
        )
        if used is not None:
            usage = add_usage(usage or {}, used)
    reflected = False
    if tokens(observations_of(log)) > reflect_tokens():
        log, used = reflect(log)
        usage = add_usage(usage or {}, used)
        reflected = True
    # No model call (a render, or a compaction whose window the background observer had already covered): no usage
    # at all, rather than an empty record pi would try to read a cost out of.
    return {
        "log": log,
        "task": task,
        "summary": render(log, task),
        "usage": usage,
        "reflected": reflected,
    }
