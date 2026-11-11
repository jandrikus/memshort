// pi extension: short-term observational memory. Compacts the live session the observational-memory way
// (Mastra's Observational Memory, pi-observational-memory): older messages are replaced by a dated observation log
// that deepseek-flash writes, and the newest messages stay as they are.
//
// - After a completed user turn, compact at compact_after_turn_tokens. During a run, compact at
//   compact_during_turn_tokens between tool batches, so a long run can retain more raw context without being aborted.
//   The footer shows the limit that applies now: the during-turn one while a run is going, the after-turn one idle.
// - Before that, every time observe_every_tokens of messages older than the kept tail have gone unobserved, they are
//   observed in the background and the log is saved as an `memshort-om` custom entry; the compaction itself then
//   only observes the little that is left.
// - /compact, and pi's own threshold or overflow recovery, get the same log through session_before_compact.
//   When memshort fails, the compaction is cancelled with the reason; pi's own summary never stands in.
// `memshort om` does the model work, shared with the Claude Code mod (mod/hooks/compact.tsx).
import { execFile } from "node:child_process";
import { realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Usage } from "@earendil-works/pi-ai";
import type {
	CompactionEntry,
	ExtensionAPI,
	ExtensionContext,
	ProjectedSessionEntry,
	SessionBoundaryDraft,
} from "@earendil-works/pi-coding-agent";
import { convertToLlm, estimateTokens, serializeConversation } from "@earendil-works/pi-coding-agent";

const bin = process.env.MEMSHORT_COMPACT_BIN ?? "memshort";
const role = process.env.MEMSHORT_ROLE ?? "interactive";
// After a failed compaction, wait this long before the next try instead of failing on every turn.
const RETRY_MS = 120_000;
const CHECKPOINT = "memshort-om";
// A checkpoint written before the rename still marks where the observed part ends.
const LEGACY_CHECKPOINT = "aimem-om";

// pi opens this file through a link in ~/.pi/agent/extensions, and jiti resolves a relative import against the
// link's directory rather than this checkout: the real file is resolved first, then the helper beside it.
async function installation(): Promise<typeof import("./lib/installation.ts")> {
	const here = dirname(realpathSync(fileURLToPath(import.meta.url)));
	return import(join(here, "lib/installation.ts"));
}
async function preservation(): Promise<typeof import("../claude/hooks/preserved.ts")> {
	const here = dirname(realpathSync(fileURLToPath(import.meta.url)));
	return import(join(here, "../claude/hooks/preserved.ts"));
}

type Checkpoint = { through: string; log: string; task: string };
type Om = { log: string; task: string; summary: string; usage?: Usage | null };
// The session as the model sees it: `start` is the first raw entry after the newest compaction, `through` the
// first entry the log does not cover yet.
// `ratio` turns the estimate into real tokens (calibration).
type View = { entries: ProjectedSessionEntry[]; start: number; through: number; log: string; task: string; ratio: number };

function om(request: { log: string; task: string; transcript: Part[]; focus?: string }, signal?: AbortSignal): Promise<Om> {
	return new Promise((resolve, reject) => {
		const child = execFile(bin, ["om"], { timeout: 600_000, maxBuffer: 16 << 20, signal }, (err, stdout, stderr) => {
			let parsed: (Om & { error?: string }) | undefined;
			try {
				parsed = JSON.parse(stdout);
			} catch {}
			if (parsed && !parsed.error && !err) resolve(parsed);
			else reject(new Error(parsed?.error ?? (stderr.trim().slice(-500) || err?.message || "no output")));
		});
		child.stdin?.end(JSON.stringify(request));
	});
}

// pi's message types leave "system" out of their roles, but a session keeps the system prompt as a `system` message
// entry at runtime, and it is sent either way, so it is not conversation to compact: count it out of the
// conversation and never cut there. The comparison is widened on purpose; the exclusion below is the whole point.
const isSystem = (role: string): boolean => role === "system";

// pi's estimate is characters / 4; on deepseek-flash the real count ran about 1.25 times that.
const estimate = (entries: ProjectedSessionEntry[]) =>
	entries.reduce(
		(n, e) => n + e.messages.reduce((m, message) => m + (isSystem(message.role) ? 0 : estimateTokens(message)), 0),
		0,
	);

// The estimated growth between the first and last reported input totals before the ratio is trusted.
const CALIBRATE_MIN = 2_000;

// Real tokens per estimated one: each assistant message carries the input total its request was answered over, so
// the growth between the first and the last of them since the newest compaction, against the estimate's growth,
// measures what characters / 4 gets wrong for this model. 1 until there is enough growth to measure.
function calibration(entries: ProjectedSessionEntry[], start: number): number {
	let estimated = 0;
	let first: { tokens: number; estimated: number } | undefined;
	let last = first;
	for (const e of entries.slice(start)) {
		for (const message of e.messages) {
			if (isSystem(message.role)) continue;
			if (message.role === "assistant") {
				const u = message.usage;
				const tokens = u.input + u.cacheRead + u.cacheWrite;
				if (tokens > 0) {
					if (!first || tokens < (last?.tokens ?? 0)) first = { tokens, estimated };
					last = { tokens, estimated };
				}
			}
			estimated += estimateTokens(message);
		}
	}
	if (!first || !last || last.estimated - first.estimated < CALIBRATE_MIN) return 1;
	return (last.tokens - first.tokens) / (last.estimated - first.estimated);
}

// A tool result stays with the call it answers, so a kept tail never starts with one.
const canCut = (e: ProjectedSessionEntry) =>
	e.messages.length > 0 && e.messages[0].role !== "toolResult" && !isSystem(e.messages[0].role);

// The first entry to keep: the newest entries worth at least `keep` tokens, from a valid cut. `from` when nothing
// older than that is left.
function cutIndex(entries: ProjectedSessionEntry[], from: number, keep: number, ratio: number): number {
	let kept = 0;
	for (let i = entries.length - 1; i > from; i--) {
		kept += estimate([entries[i]]) * ratio;
		if (kept >= keep && canCut(entries[i])) return i;
	}
	return from;
}

type Lib = typeof import("../claude/hooks/preserved.ts");
type Message = ProjectedSessionEntry["messages"][number];
type Part = import("../claude/hooks/preserved.ts").Part;

const SKILL_BLOCK = /^<skill name="([^"]+)" location="[^"]+">\n/;

const textOf = (content: string | readonly { type: string; text?: string }[]) =>
	typeof content === "string"
		? content
		: content
				.filter((block) => block.type === "text")
				.map((block) => block.text ?? "")
				.join("\n");

// The slice as ordered transcript parts for `memshort om`: conversation runs, and between them the text kept verbatim
// in the log. That is the first user message when the slice starts the session (`first`), a `/skill:name` expansion
// (pi's <skill> block) and the result of a successful `read` of a SKILL.md. The text of a kept message is not also
// sent as conversation, and an image in the first user message is noted for the observer.
function transcriptParts(lib: Lib, entries: ProjectedSessionEntry[], first: boolean): Part[] {
	const parts = new lib.Parts<Message>((messages) => serializeConversation(convertToLlm(messages)));
	const reads = new Map<string, string>();
	for (const message of entries.flatMap((e) => e.messages)) {
		if (isSystem(message.role)) continue;
		if (message.role === "user") {
			const text = textOf(message.content);
			const skill = SKILL_BLOCK.exec(text);
			if (skill) {
				parts.keep({ kind: "skill", name: skill[1], content: text });
				first = false;
				continue;
			}
			if (first) {
				first = false;
				const images = typeof message.content === "string" ? 0 : message.content.filter((b) => b.type === "image").length;
				parts.keep({ kind: "user", content: text });
				if (images) parts.text({ ...message, content: [{ type: "text", text: `(the first user message also carried ${images} image(s), not kept)` }] });
				continue;
			}
		} else if (message.role === "assistant") {
			for (const block of message.content)
				if (block.type === "toolCall" && block.name === "read") {
					const path = (block.arguments as { path?: unknown } | undefined)?.path;
					if (typeof path === "string" && lib.SKILL_FILE.test(path)) reads.set(block.id, path);
				}
		} else if (message.role === "toolResult" && !message.isError && reads.has(message.toolCallId)) {
			parts.keep({ kind: "skill", name: reads.get(message.toolCallId), content: textOf(message.content) });
			continue;
		}
		parts.text(message);
	}
	return parts.done();
}

// pi reads `usage.cost.total` with no guard when it notices a compaction's cost and in the footer's totals, so a
// usage record without a cost object crashes the harness (pi 1.4.0). `memshort om` returns null when it made no model
// call; an older memshort may still hand over `{}`, which is why the shape is checked here too.
const billable = (usage: Om["usage"]): Usage | undefined =>
	usage && typeof usage.cost?.total === "number" ? usage : undefined;

function view(ctx: ExtensionContext): View {
	const entries = ctx.sessionManager.buildSessionProjection().entries;
	// The newest compaction is projected first; older ones in its kept range contribute no messages.
	const at = entries.findIndex((e) => e.sourceEntry.type === "compaction" && e.messages.length > 0);
	const start = at + 1;
	const ratio = calibration(entries, start);
	let log = "";
	let task = "";
	if (at >= 0) {
		const compaction = entries[at].sourceEntry as CompactionEntry<Partial<Checkpoint>>;
		if (compaction.fromHook && typeof compaction.details?.log === "string") {
			log = compaction.details.log;
			task = compaction.details.task ?? "";
		} else {
			// pi's own summary seeds the log. It is model text, so it must not open a preserved block.
			log = compaction.summary.replaceAll("<memshort-preserved>", "‹memshort-preserved›");
		}
	}
	// The newest checkpoint since that compaction whose boundary is still on this branch.
	const index = new Map(entries.map((e, i) => [e.sourceEntry.id, i]));
	const branch = ctx.sessionManager.getBranch();
	const compactionId = at >= 0 ? entries[at].sourceEntry.id : undefined;
	for (let i = branch.length - 1; i >= 0; i--) {
		const entry = branch[i];
		if (entry.id === compactionId) break;
		if (entry.type !== "custom" || (entry.customType !== CHECKPOINT && entry.customType !== LEGACY_CHECKPOINT)) continue;
		const checkpoint = entry.data as Checkpoint;
		const through = index.get(checkpoint.through);
		if (through !== undefined && through >= start)
			return { entries, start, through, log: checkpoint.log, task: checkpoint.task, ratio };
	}
	return { entries, start, through: start, log, task, ratio };
}

export default async function (pi: ExtensionAPI) {
	if (role !== "interactive" || process.env.MEMSHORT_COMPACT === "off") return;
	// The selection is this host's: ~/.pi is shared between machines, so the link alone cannot say what runs here.
	// Compaction is all this extension does; the host record and the memory tools belong to the memory extension.
	const lib = await installation();
	const preserved = await preservation();
	const gate = lib.featureGate("compact", "pi");
	const off = (error: string) =>
		pi.on("session_start", (_event, ctx) => ctx.ui.notify(`memshort: the compact feature is off: ${error}`, "error"));
	if (!gate.enabled) {
		if (gate.reason === "invalid") off(gate.error);
		return;
	}
	// Raw tokens after the last compaction that start the next one, the newest raw tokens a compaction keeps, and
	// the unobserved tokens older than that which start a background observation. The reflector's limit is read by
	// `memshort om` itself.
	let limits: ReturnType<typeof lib.thresholds>;
	try {
		limits = lib.thresholds(gate.installation);
	} catch (err) {
		off((err as Error).message);
		return;
	}
	const AFTER_TURN = limits.compact_after_turn_tokens;
	const DURING_TURN = limits.compact_during_turn_tokens;
	const KEEP = limits.keep_recent_tokens;
	const OBSERVE_EVERY = limits.observe_every_tokens;

	let inflight: Promise<void> | undefined;
	let generation = 0; // a background result from before a session switch belongs to the old session
	let failedAt = 0;
	let phase = "idle";
	let running = false; // between agent_start and agent_settled the during-turn limit applies

	function status(ctx: ExtensionContext) {
		const v = view(ctx);
		const round = ctx.sessionManager.getBranch().filter((e) => e.type === "compaction").length;
		const raw = estimate(v.entries.slice(v.start)) * v.ratio;
		const count = preserved.observationsOf(v.log).match(/^[*-]\s+[🔴🟡🟢]/gmu)?.length ?? 0;
		const fmt = (n: number) => `${(n / 1000).toFixed(1)}k`;
		const limit = running ? DURING_TURN : AFTER_TURN;
		ctx.ui.setStatus("memshort-om", `OM r${round} · raw ~${fmt(raw)}/${fmt(limit)} · ${count} obs · log ~${fmt(v.log.length / 3.5)} · ${phase}`);
	}

	function switched(ctx: ExtensionContext) {
		generation++;
		failedAt = 0;
		phase = "idle";
		running = false;
		status(ctx);
	}
	pi.on("session_start", (_event, ctx) => switched(ctx));
	pi.on("session_tree", (_event, ctx) => switched(ctx));
	pi.on("session_compact", (_event, ctx) => status(ctx));
	pi.on("agent_start", (_event, ctx) => {
		running = true;
		status(ctx);
	});
	pi.on("agent_settled", (_event, ctx) => {
		running = false;
		status(ctx);
	});
	pi.on("agent_end", (_event, ctx) => status(ctx));
	pi.on("message_end", (_event, ctx) => status(ctx));
	pi.on("session_shutdown", (_event, ctx) => {
		generation++;
		ctx.ui.setStatus("memshort-om", undefined);
	});

	function buffer(ctx: ExtensionContext) {
		if (inflight) return;
		const v = view(ctx);
		const cut = cutIndex(v.entries, v.through, KEEP, v.ratio);
		const pending = v.entries.slice(v.through, cut);
		if (estimate(pending) * v.ratio < OBSERVE_EVERY) return;
		const through = v.entries[cut].sourceEntry.id;
		const started = generation;
		phase = "observing";
		status(ctx);
		// After a session switch or shutdown, ctx is stale and the result belongs to the old session: drop it.
		inflight = om({ log: v.log, task: v.task, transcript: transcriptParts(preserved, pending, v.through === 0) })
			.then((r) => {
				if (started === generation) pi.appendEntry<Checkpoint>(CHECKPOINT, { through, log: r.log, task: r.task });
			})
			.catch((err: Error) => {
				if (started === generation) ctx.ui.notify(`memshort: background observation failed: ${err.message}`, "warning");
			})
			.finally(() => {
				inflight = undefined;
				if (started === generation) {
					phase = "idle";
					status(ctx);
				}
			});
	}

	// The compaction up to `keepFrom` (an entry id), or up to the KEEP tail; undefined when nothing is older.
	async function activate(ctx: ExtensionContext, focus?: string, keepFrom?: string, signal?: AbortSignal) {
		const started = generation;
		await inflight;
		if (started !== generation) return undefined;
		const v = view(ctx);
		const cut = keepFrom ? v.entries.findIndex((e) => e.sourceEntry.id === keepFrom) : cutIndex(v.entries, v.start, KEEP, v.ratio);
		if (cut <= v.start) return undefined;
		const pending = v.through < cut ? v.entries.slice(v.through, cut) : [];
		phase = "compacting";
		status(ctx);
		try {
			const parts = pending.length ? transcriptParts(preserved, pending, v.through === 0) : [];
			const r = await om({ log: v.log, task: v.task, transcript: parts, focus }, signal);
			if (started !== generation) return undefined;
			return {
				summary: r.summary,
				firstKeptEntryId: v.entries[cut].sourceEntry.id,
				details: { log: r.log, task: r.task },
				usage: billable(r.usage),
			};
		} finally {
			if (started === generation) {
				phase = "idle";
				status(ctx);
			}
		}
	}

	async function atBoundary(ctx: ExtensionContext, entries: SessionBoundaryDraft[], threshold: number) {
		const v = view(ctx);
		if (estimate(v.entries.slice(v.start)) * v.ratio < threshold || Date.now() - failedAt < RETRY_MS) {
			buffer(ctx);
			return;
		}
		try {
			const compaction = await activate(ctx, undefined, undefined, ctx.signal);
			if (!compaction) return;
			return { entries: [...entries, { type: "compaction" as const, ...compaction }] };
		} catch (err) {
			failedAt = Date.now();
			ctx.ui.notify(`memshort: compaction failed, retrying in ${RETRY_MS / 60_000} min: ${(err as Error).message}`, "warning");
		}
	}

	pi.on("turn_end", (event, ctx) => atBoundary(ctx, event.entries, DURING_TURN));
	pi.on("agent_before_settle", (event, ctx) => {
		running = false; // the turn is complete; a continuation starts again at agent_start
		return atBoundary(ctx, event.entries, AFTER_TURN);
	});

	pi.on("session_before_compact", async (event, ctx) => {
		// An error, never pi's own summary in its place: the compaction is cancelled and the reason shown.
		try {
			const compaction = await activate(ctx, event.customInstructions, event.preparation.firstKeptEntryId, event.signal);
			if (compaction) return { compaction: { ...compaction, tokensBefore: event.preparation.tokensBefore } };
			ctx.ui.notify("memshort: nothing older than the newest messages to compact", "error");
		} catch (err) {
			ctx.ui.notify(`memshort: compaction failed: ${(err as Error).message}`, "error");
		}
		return { cancel: true };
	});
}
