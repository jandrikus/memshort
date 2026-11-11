// Host-local memshort selection. Shared Pi settings may load this file on multiple hosts.
// The gate fails closed when the selection is missing or malformed, so only a host that opted in compacts.
// The Python installer writes MEMSHORT_COMPACT_INSTALL_CONFIG or the memshort file under XDG config home.
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export type Feature = "compact" | "memory";
export type Harness = "claude" | "codex" | "pi" | "hermes";

export const VERSION = 2;
// Version 1 named pi models without the `pi/` prefix; the rest of the file is the same, so both are read.
export const READS: readonly number[] = [1, 2];
export const FEATURES: readonly Feature[] = ["compact"];
export const HARNESSES: readonly Harness[] = ["claude", "codex", "pi", "hermes"];
// What each harness can run: Hermes is a plugin on its context-engine interface, so compaction only.
export const HARNESS_FEATURES: Record<Harness, readonly Feature[]> = {
	claude: FEATURES,
	pi: FEATURES,
	codex: ["compact"],
	hermes: ["compact"],
};
// Without a saved selection a host keeps what it always ran; Hermes is not in it, since no host asked for it by
// default and a plain `memshort install` must not touch it.
export const DEFAULT_FEATURES: readonly Feature[] = FEATURES;
export const DEFAULT_HARNESSES: readonly Harness[] = ["claude", "pi"];
export const DEFAULT_MODEL = "deepseek/deepseek-flash";
export const ENV_PATH = "MEMSHORT_COMPACT_INSTALL_CONFIG";
export const ENV_XDG = "XDG_CONFIG_HOME";
const KEYS = ["version", "features", "harnesses", "model", "observer_model", "reflector_model", "thresholds", "endpoints"];
// The same table as THRESHOLDS in installation.py: file key, environment variable, default tokens.
export const THRESHOLDS = {
	compact_after_turn_tokens: ["MEMSHORT_COMPACT_AFTER_TURN_TOKENS", 100_000],
	compact_during_turn_tokens: ["MEMSHORT_COMPACT_DURING_TURN_TOKENS", 200_000],
	keep_recent_tokens: ["MEMSHORT_KEEP_RECENT_TOKENS", 15_000],
	observe_every_tokens: ["MEMSHORT_OBSERVE_EVERY_TOKENS", 10_000],
	condense_log_above_tokens: ["MEMSHORT_CONDENSE_LOG_ABOVE_TOKENS", 60_000],
} as const satisfies Record<string, readonly [string, number]>;
export type Threshold = keyof typeof THRESHOLDS;
// `<endpoint>/<model-id>`, e.g. deepseek/deepseek-flash or pi/anthropic/claude-haiku-4-5. `memshort om` resolves the
// endpoint itself (backends.py); here the shape is checked so a broken file turns the gate off with its reason. The
// endpoint is one path segment, the id may carry more slashes and `:` versions, and nothing starts with `-`.
const MODEL_SPEC = /^[A-Za-z0-9_][A-Za-z0-9._-]*\/[^\s\u0000-\u001f\u007f-][^\s\u0000-\u001f\u007f]*$/;

/** A selection that cannot be used: an unknown name, an empty list, or a file that does not parse. */
export class InstallationError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "InstallationError";
	}
}

/** One install selection: the features, the harnesses and the compaction model. */
export type Installation = {
	features: ReadonlySet<Feature>;
	harnesses: ReadonlySet<Harness>;
	model: string;
	/** Only the thresholds set on this host; `thresholds()` fills in the environment and the defaults. */
	thresholds: Partial<Record<Threshold, number>>;
};

/** The environment parts the path is built from: `process.env` in an extension, a literal in a test. */
export type InstallationEnv = { MEMSHORT_COMPACT_INSTALL_CONFIG?: string; XDG_CONFIG_HOME?: string; HOME?: string };

/** Where the selection is saved: the test override, else the XDG config home, else `~/.config`. */
export function configPath(env: InstallationEnv = process.env): string {
	const home = env.HOME || homedir();
	const override = env.MEMSHORT_COMPACT_INSTALL_CONFIG;
	if (override) return expandHome(override, home);
	const base = env.XDG_CONFIG_HOME;
	return join(base ? expandHome(base, home) : join(home, ".config"), "memshort", "install.json");
}

/** The saved selection; an absent file is the defaults, anything malformed is an error that names the file. */
export function loadInstallation(env: InstallationEnv = process.env): Installation {
	const path = configPath(env);
	let text: string;
	try {
		text = readFileSync(path, "utf8");
	} catch (err) {
		const code = (err as NodeJS.ErrnoException).code;
		if (code === "ENOENT") return selection(DEFAULT_FEATURES, DEFAULT_HARNESSES, DEFAULT_MODEL);
		throw new InstallationError(`cannot read ${path}: ${(err as Error).message}`);
	}
	return parseInstallation(text, path);
}

/** One file's contents as a selection; every problem is an `InstallationError` that names the file. */
export function parseInstallation(text: string, path: string): Installation {
	let data: unknown;
	try {
		data = JSON.parse(text);
	} catch (err) {
		throw new InstallationError(`${path} is not valid JSON: ${(err as Error).message}`);
	}
	if (data === null || typeof data !== "object" || Array.isArray(data))
		throw new InstallationError(`${path} must hold a JSON object`);
	const record = data as Record<string, unknown>;
	const unknown = Object.keys(record)
		.filter((key) => !KEYS.includes(key))
		.sort();
	if (unknown.length)
		throw new InstallationError(`${path} has unknown keys: ${unknown.join(", ")}; known: ${KEYS.join(", ")}`);
	const version = "version" in record ? record.version : VERSION;
	if (typeof version !== "number" || !READS.includes(version))
		throw new InstallationError(`${path} is version ${show(version)}, this memshort reads versions ${READS.join(", ")}`);
	const endpoints = record.endpoints ?? {};
	if (endpoints === null || typeof endpoints !== "object" || Array.isArray(endpoints))
		throw new InstallationError(`${path}: endpoints must be an object, not ${show(endpoints)}`);
	const missing = ["features", "harnesses"].filter((key) => !(key in record));
	if (missing.length) throw new InstallationError(`${path} is missing ${missing.join(", ")}`);
	for (const key of ["features", "harnesses"])
		if (!Array.isArray(record[key]))
			throw new InstallationError(`${path}: ${key} must be a list, not ${show(record[key])}`);
	// `memshort om` resolves the stage models itself; the extension only checks the file stays usable.
	for (const key of ["observer_model", "reflector_model"])
		if (key in record) {
			try {
				modelName(record[key]);
			} catch (err) {
				throw new InstallationError(`${path}: ${key}: ${(err as Error).message}`);
			}
		}
	try {
		return selection(
			record.features,
			record.harnesses,
			record.model === undefined ? DEFAULT_MODEL : record.model,
			record.thresholds === undefined ? {} : record.thresholds,
		);
	} catch (err) {
		if (!(err instanceof InstallationError)) throw err;
		throw new InstallationError(`${path}: ${err.message}`);
	}
}

/** A checked selection, from the same names the Python side accepts; the argument order is its `Selection`. */
export function selection(
	features: unknown = DEFAULT_FEATURES,
	harnesses: unknown = DEFAULT_HARNESSES,
	model: unknown = DEFAULT_MODEL,
	thresholds: unknown = {},
): Installation {
	const chosen = names(features, FEATURES, "feature");
	const chosenHarnesses = names(harnesses, HARNESSES, "harness");
	const chosenModel = modelName(model);
	const own = ownThresholds(thresholds);
	carries(chosen, chosenHarnesses);
	return { features: new Set(chosen), harnesses: new Set(chosenHarnesses), model: chosenModel, thresholds: own };
}

/**
 * Every threshold in effect: its environment variable, else the file, else the default. A bad environment value
 * is an `InstallationError` that names the variable, never a silent default.
 */
export function thresholds(
	installation: Installation,
	env: Record<string, string | undefined> = process.env,
): Record<Threshold, number> {
	const out = {} as Record<Threshold, number>;
	for (const [key, [variable, fallback]] of Object.entries(THRESHOLDS) as [Threshold, readonly [string, number]][]) {
		const raw = env[variable]?.trim();
		out[key] = raw ? tokenCount(raw, variable) : (installation.thresholds[key] ?? fallback);
	}
	return out;
}

function ownThresholds(value: unknown): Partial<Record<Threshold, number>> {
	if (value === null || typeof value !== "object" || Array.isArray(value))
		throw new InstallationError(`thresholds must be an object, not ${show(value)}`);
	const record = value as Record<string, unknown>;
	const unknown = Object.keys(record)
		.filter((key) => !(key in THRESHOLDS))
		.sort();
	if (unknown.length)
		throw new InstallationError(
			`unknown thresholds: ${unknown.join(", ")}; known: ${Object.keys(THRESHOLDS).join(", ")}`,
		);
	const out: Partial<Record<Threshold, number>> = {};
	for (const [key, raw] of Object.entries(record)) out[key as Threshold] = tokenCount(raw, key);
	return out;
}

/** A positive whole number of tokens, as `_token_count` in installation.py accepts it. */
function tokenCount(value: unknown, name: string): number {
	let n = value;
	if (typeof n === "string") {
		const text = n.trim().replace(/_/g, "");
		if (/^\d+$/.test(text)) n = Number(text);
	}
	if (typeof n !== "number" || !Number.isSafeInteger(n) || n <= 0)
		throw new InstallationError(`${name} must be a positive whole number of tokens, not ${show(value)}`);
	return n;
}

/**
 * Whether `harness` runs `feature` on this host.
 *
 * `invalid` carries the reason a malformed file disables the feature instead of the defaults enabling it; a caller
 * shows it and runs nothing. `not-selected` is a host that simply does not want the feature, and says nothing.
 */
export type Gate =
	| { enabled: true; installation: Installation }
	| { enabled: false; reason: "not-selected" }
	| { enabled: false; reason: "invalid"; error: string };

/** The gate for one feature in one harness, read from disk on every call so a reload picks up a new selection. */
export function featureGate(feature: Feature, harness: Harness, env: InstallationEnv = process.env): Gate {
	let installation: Installation;
	const path = configPath(env);
	let text: string;
	try {
		text = readFileSync(path, "utf8");
	} catch (err) {
		// Shared Pi settings can load this extension on a host that did not install it.
		if ((err as NodeJS.ErrnoException).code === "ENOENT") return { enabled: false, reason: "not-selected" };
		return { enabled: false, reason: "invalid", error: `cannot read ${path}: ${(err as Error).message}` };
	}
	try {
		installation = parseInstallation(text, path);
	} catch (err) {
		if (!(err instanceof InstallationError)) throw err;
		return { enabled: false, reason: "invalid", error: err.message };
	}
	const enabled =
		installation.harnesses.has(harness) &&
		installation.features.has(feature) &&
		HARNESS_FEATURES[harness].includes(feature);
	return enabled ? { enabled: true, installation } : { enabled: false, reason: "not-selected" };
}

/** `~` and `~/...` as this environment's home; Python's `expanduser` for the paths above. */
function expandHome(path: string, home: string): string {
	if (path === "~") return home;
	if (path.startsWith("~/")) return join(home, path.slice(2));
	return path;
}

/** Sorted, de-duplicated names from `allowed`; a comma or whitespace separated string is accepted too. */
function names<T extends string>(value: unknown, allowed: readonly T[], kind: string): T[] {
	const given = typeof value === "string" ? value.replace(/,/g, " ").split(/\s+/).filter(Boolean) : value;
	if (!Array.isArray(given))
		throw new InstallationError(`the ${kind} selection must be a string or a list, not ${show(value)}`);
	const chosen: T[] = [];
	for (const name of given) {
		if (typeof name !== "string") throw new InstallationError(`${kind} names must be strings, not ${show(name)}`);
		const key = name.trim().toLowerCase();
		if (!allowed.includes(key as T))
			throw new InstallationError(`unknown ${kind} ${show(name)}; choose from ${allowed.join(", ")}`);
		if (chosen.includes(key as T)) throw new InstallationError(`${kind} ${show(key)} is selected twice`);
		chosen.push(key as T);
	}
	if (!chosen.length) throw new InstallationError(`select at least one ${kind}: ${allowed.join(", ")}`);
	return chosen.sort();
}

/** Refuse a selection some harness cannot run, or a feature no selected harness carries. */
function carries(features: Feature[], harnesses: Harness[]): void {
	for (const harness of harnesses) {
		if (features.some((feature) => HARNESS_FEATURES[harness].includes(feature))) continue;
		if (harness === "hermes")
			throw new InstallationError(
				"Hermes has no long-term memory support: hermes needs the compact feature, or leave hermes out",
			);
		throw new InstallationError(`${harness} runs only the ${HARNESS_FEATURES[harness].join(", ")} feature`);
	}
	for (const feature of features) {
		if (harnesses.some((harness) => HARNESS_FEATURES[harness].includes(feature))) continue;
		if (feature === "memory" && harnesses.includes("hermes"))
			throw new InstallationError(
				"Hermes has no long-term memory support: the memory feature needs claude or pi, so select one of them",
			);
		throw new InstallationError(`no selected harness runs the ${feature} feature`);
	}
}

function modelName(value: unknown): string {
	if (typeof value !== "string" || !MODEL_SPEC.test(value.trim()))
		throw new InstallationError(`model must look like provider/model, e.g. ${DEFAULT_MODEL}, not ${show(value)}`);
	return value.trim();
}

/** A value as the Python side's `!r` reads: strings in single quotes, everything else as JSON. */
function show(value: unknown): string {
	if (typeof value === "string") return `'${value}'`;
	try {
		return JSON.stringify(value) ?? String(value);
	} catch {
		return String(value);
	}
}
