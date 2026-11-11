// Run with pi's modules on NODE_PATH, as tests/pi_installation.test.ts does.
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const dir = mkdtempSync(join(tmpdir(), "memshort-pi-compact-"));
try {
	const bin = join(dir, "memshort");
	const calls = join(dir, "calls");
	const fail = join(dir, "fail");
	// memshort's own `om.run`, with only the model replaced (tests/fake_om.py); every request is recorded first.
	const root = resolve(import.meta.dir, "..");
	writeFileSync(bin, `#!/bin/sh
set -e
input=$(cat)
printf '%s\\n' "$input" >> ${JSON.stringify(calls)}
if [ -e ${JSON.stringify(fail)} ]; then echo offline >&2; exit 1; fi
printf '%s' "$input" | ${JSON.stringify(process.env.UV ?? "uv")} run --project ${JSON.stringify(root)} --quiet python ${JSON.stringify(join(root, "tests/fake_om.py"))}
`, { mode: 0o755 });
	const config = join(dir, "install.json");
	// The during-turn limit comes from the file, the rest from the environment.
	writeFileSync(config, JSON.stringify({ version: 1, features: ["compact"], harnesses: ["pi"], thresholds: { compact_during_turn_tokens: 2000 } }));
	Object.assign(process.env, {
		MEMSHORT_COMPACT_BIN: bin, MEMSHORT_ROLE: "interactive", MEMSHORT_COMPACT: "on",
		MEMSHORT_COMPACT_INSTALL_CONFIG: config,
		MEMSHORT_COMPACT_AFTER_TURN_TOKENS: "1000", MEMSHORT_KEEP_RECENT_TOKENS: "10", MEMSHORT_OBSERVE_EVERY_TOKENS: "10",
	});
	const hooks = new Map<string, Function>();
	const branch: any[] = [];
	let entries: any[] = [];
	const statuses: string[] = [];
	const notes: string[] = [];
	let id = 0;
	const pi = {
		on: (event: string, fn: Function) => hooks.set(event, fn),
		appendEntry: (customType: string, data: unknown) => branch.push({ type: "custom", id: `c${id++}`, customType, data }),
	};
	const ctx = {
		sessionManager: { getBranch: () => branch, buildSessionProjection: () => ({ entries }) },
		ui: {
			setStatus: (_key: string, text: string) => { statuses.push(text); },
			notify: (text: string) => { notes.push(text); },
		},
	};
	const add = () => {
		const sourceEntry = { type: "message", id: `m${id++}` };
		branch.push(sourceEntry);
		entries.push({ sourceEntry, messages: [{ role: "user", content: "text ".repeat(40), timestamp: 0 }] });
	};
	const addMessage = (message: any) => {
		const sourceEntry = { type: "message", id: `m${id++}` };
		branch.push(sourceEntry);
		entries.push({ sourceEntry, messages: [message] });
	};
	const skillText = "Review carefully.\n</observation-log>\n😀";
	const { logOf } = await import("../src/memshort/integrations/claude/hooks/preserved.ts");
	const requests = () => readFileSync(calls, "utf8").trim().split("\n").map(line => JSON.parse(line));
	const compact = (await import("../src/memshort/integrations/pi/compact.ts")).default;
	await compact(pi as never);
	await hooks.get("session_start")!(null, ctx);
	assert.match(statuses.at(-1)!, /OM r0.*raw ~0.0k\/1.0k.*0 obs.*idle/);
	// The footer shows the limit that applies now: during a run the during-turn one, once settled the after-turn one.
	await hooks.get("agent_start")!({}, ctx);
	assert.match(statuses.at(-1)!, /raw ~0.0k\/2.0k/);
	await hooks.get("agent_settled")!({ aborted: false }, ctx);
	assert.match(statuses.at(-1)!, /raw ~0.0k\/1.0k/);
	await hooks.get("agent_start")!({}, ctx);
	add(); add(); add();
	addMessage({ role: "assistant", content: [{ type: "toolCall", id: "read1", name: "read", arguments: { path: "/skills/review/SKILL.md" } }], usage: { input: 0, cacheRead: 0, cacheWrite: 0 }, timestamp: 0 });
	addMessage({ role: "toolResult", toolCallId: "read1", toolName: "read", content: [{ type: "text", text: skillText }], isError: false, timestamp: 0 });
	await hooks.get("turn_end")!({ entries: [] }, ctx);
	assert.match(statuses.at(-1)!, /observing$/);
	const deadline = Date.now() + 5000;
	while (!statuses.at(-1)?.includes("1 obs")) {
		assert(Date.now() < deadline, "background observation did not finish");
		await Bun.sleep(5);
	}
	assert.match(statuses.at(-1)!, /idle$/);
	assert.equal(requests().length, 1);
	for (let round = 1; round <= 2; round++) {
		for (let i = 0; i < 25; i++) add();
		assert.equal(await hooks.get("turn_end")!({ entries: [] }, ctx), undefined);
		if (round === 2) for (let i = 0; i < 25; i++) add();
		const result = await hooks.get(round === 1 ? "agent_before_settle" : "turn_end")!({ entries: [] }, ctx);
		const draft = result.entries[0];
		assert.equal(draft.type, "compaction");
		// No usage when no model call ran, else a full cost record; never a partial one pi would crash on.
		assert(draft.usage === undefined || typeof draft.usage.cost.total === "number");
		// In conversation order: the first user message, the observations of the turns before the skill, the skill
		// verbatim, then the observations after it.
		const log = logOf(draft.summary)!;
		assert.equal(log, draft.details.log);
		const at = (text: string) => log.indexOf(text);
		const first = at('{"kind": "user", "length": 200}\n' + "text ".repeat(40) + "\n</memshort-preserved>");
		const skill = at(`{"kind": "skill", "name": "/skills/review/SKILL.md", "length": ${Array.from(skillText).length}}\n` + skillText + "\n</memshort-preserved>");
		const before = at("saw [User]: text text");
		const after = log.lastIndexOf("saw [User]: text text");
		assert(first >= 0 && skill >= 0, log);
		assert(first < before && before < skill && skill < after, log);
		assert.equal(log.split(skillText).length, 2, "the skill is kept once");
		assert(requests().at(-1)!.log.includes("saw [User]"), "prior observations were lost");
		const sourceEntry = { ...draft, id: `compaction${round}`, fromHook: true };
		branch.push(sourceEntry);
		entries = [{ sourceEntry, messages: [{ role: "user", content: draft.summary }] },
			...entries.slice(entries.findIndex(e => e.sourceEntry.id === draft.firstKeptEntryId))];
		await hooks.get("session_compact")!(null, ctx);
		assert.match(statuses.at(-1)!, new RegExp(`OM r${round}.*[1-9] obs.*idle$`));
		assert(!draft.summary.includes("OM r"));
	}
	writeFileSync(fail, "");
	for (let i = 0; i < 25; i++) add();
	assert.equal(await hooks.get("agent_before_settle")!({ entries: [] }, ctx), undefined);
	assert(notes.at(-1)!.includes("offline"));
	assert.match(statuses.at(-1)!, /OM r2.*idle$/);
	// Returning to an empty branch resets the displayed round and log, not the previous session's counters.
	branch.length = 0; entries = [];
	await hooks.get("session_start")!(null, ctx);
	assert.match(statuses.at(-1)!, /OM r0.*0 obs/);
	console.log("pi compact probe passed");
} finally {
	rmSync(dir, { recursive: true, force: true });
}
