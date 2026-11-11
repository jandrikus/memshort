import type { EngineInterface } from 'claude-code'

// The same table as THRESHOLDS in installation.py: file key, environment variable, default tokens.
export const THRESHOLDS = {
  compact_after_turn_tokens: ['MEMSHORT_COMPACT_AFTER_TURN_TOKENS', 100_000],
  compact_during_turn_tokens: ['MEMSHORT_COMPACT_DURING_TURN_TOKENS', 200_000],
  keep_recent_tokens: ['MEMSHORT_KEEP_RECENT_TOKENS', 15_000],
  observe_every_tokens: ['MEMSHORT_OBSERVE_EVERY_TOKENS', 10_000],
  condense_log_above_tokens: ['MEMSHORT_CONDENSE_LOG_ABOVE_TOKENS', 60_000],
} as const satisfies Record<string, readonly [string, number]>
export type Threshold = keyof typeof THRESHOLDS
export type Selected = { features: Set<string>; thresholds: Partial<Record<Threshold, number>> }

/** A positive whole number of tokens, as `_token_count` in installation.py accepts it; undefined when it is not. */
export function tokenCount(value: unknown): number | undefined {
  let n = value
  if (typeof n === 'string') {
    const text = n.trim().replace(/_/g, '')
    if (/^\d+$/.test(text)) n = Number(text)
  }
  return typeof n === 'number' && Number.isSafeInteger(n) && n > 0 ? n : undefined
}

/** Every threshold in effect: its environment variable, else the file, else the default. A bad value throws. */
export async function thresholds($: EngineInterface, own: Selected['thresholds'] = {}): Promise<Record<Threshold, number>> {
  const out = {} as Record<Threshold, number>
  for (const [key, [variable, fallback]] of Object.entries(THRESHOLDS) as [Threshold, readonly [string, number]][]) {
    const raw = (await $.env.get(variable))?.trim()
    if (!raw) {
      out[key] = own[key] ?? fallback
      continue
    }
    const n = tokenCount(raw)
    if (n === undefined) throw new Error(`${variable} must be a positive whole number of tokens, not '${raw}'`)
    out[key] = n
  }
  return out
}

// Install choices are host-local, not part of the memory store or synced Claude settings.
export async function installedFeatures($: EngineInterface): Promise<Set<string>> {
  return (await installed($)).features
}

export async function installed($: EngineInterface): Promise<Selected> {
  const home = await $.env.get('HOME')
  const override = await $.env.get('MEMSHORT_COMPACT_INSTALL_CONFIG')
  const base = await $.env.get('XDG_CONFIG_HOME')
  if (!home && !override && !base) throw new Error('HOME is unset; cannot locate the memshort installation config')
  let path = override || `${base || `${home}/.config`}/memshort/install.json`
  if (path === '~' || path.startsWith('~/')) {
    if (!home) throw new Error('HOME is unset; cannot expand the memshort installation config path')
    path = home + path.slice(1)
  } else if (path.startsWith('~')) {
    throw new Error('use an absolute path for another user’s memshort installation config')
  }
  if (!(await $.fs.exists(path))) return { features: new Set(), thresholds: {} }
  const config = JSON.parse(await $.fs.read(path))
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error(`invalid installation config in ${path}`)
  }
  const { features, harnesses } = config
  if ((config.version !== undefined && config.version !== 1 && config.version !== 2) ||
      Object.keys(config).some(key => !['version', 'features', 'harnesses', 'model', 'observer_model', 'reflector_model', 'thresholds', 'endpoints'].includes(key)) ||
      (config.endpoints !== undefined && (!config.endpoints || typeof config.endpoints !== 'object' || Array.isArray(config.endpoints))) ||
      !Array.isArray(features) || !features.length || features.some((f: unknown) => f !== 'compact') ||
      !Array.isArray(harnesses) || !harnesses.length ||
      harnesses.some((h: unknown) => typeof h !== 'string' || !['claude', 'codex', 'pi', 'hermes'].includes(h)) ||
      new Set(features).size !== features.length || new Set(harnesses).size !== harnesses.length ||
      (harnesses.some((h: string) => h === 'hermes' || h === 'codex') && !features.includes('compact')) ||
      (features.includes('memory') && !harnesses.some((h: string) => h === 'claude' || h === 'pi')) ||
      ['model', 'observer_model', 'reflector_model'].some(key => config[key] !== undefined && (typeof config[key] !== 'string' ||
        !/^[A-Za-z0-9_][A-Za-z0-9._-]*\/[^\s\x00-\x1f\x7f]+$/.test(config[key].trim())))) {
    throw new Error(`invalid installation choices in ${path}; repair this file before running memshort install`)
  }
  const own = config.thresholds ?? {}
  if (!own || typeof own !== 'object' || Array.isArray(own)) throw new Error(`${path}: thresholds must be an object`)
  const chosen: Selected['thresholds'] = {}
  for (const [key, value] of Object.entries(own)) {
    if (!(key in THRESHOLDS)) throw new Error(`${path}: unknown threshold ${key}; known: ${Object.keys(THRESHOLDS).join(', ')}`)
    const n = tokenCount(value)
    if (n === undefined) throw new Error(`${path}: ${key} must be a positive whole number of tokens, not ${JSON.stringify(value)}`)
    chosen[key as Threshold] = n
  }
  return { features: new Set(harnesses.includes('claude') ? features : []), thresholds: chosen }
}
