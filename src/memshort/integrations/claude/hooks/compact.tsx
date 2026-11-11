// These documented state helpers are available at runtime but missing from the published 2.1.311 declarations.
// @ts-expect-error Engine declarations do not yet export atom, read and update.
import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, SessionMessage } from 'claude-code'
import { thresholds, type Selected } from './installation.ts'
import { logOf, observationsOf, Parts, SKILL_FILE, type Part } from './preserved.ts'

// Short-term observational memory for Claude Code, the same design as pi/compact.ts: older messages are replaced
// by a dated observation log that deepseek-flash writes (`memshort om`), and the newest messages stay as they are.
//
// - Every compaction (/compact, the engine's threshold, ours) is answered by the session.compact hook with the log
//   and the kept tail, the main conversation's and each subagent's. When memshort fails, the compaction fails with its
//   reason; the engine's own summary never stands in.
// - After a user turn, compact at compact_after_turn_tokens. During a turn, compact at compact_during_turn_tokens
//   or the engine's
//   own window limit. A mod can only start a compaction between turns, but the engine compacts on its own (`auto`)
//   before a request whose input passes its window less a reserve. After each main step the mod moves
//   CLAUDE_CODE_AUTO_COMPACT_WINDOW; after a turn it also queues /compact. The
//   variable is process-wide, so subagents compact at the same input total, which with their smaller system
//   prompt is somewhat later.
// - The status line shows the limit that applies now: the during-turn one while a main turn runs, else the other.
// - During and after turns, every observe_every_tokens of messages older than the kept tail are observed in the
//   background, so a compaction only observes the little that is left. Each loop (main, every subagent) keeps
//   its own checkpoint.

const OPEN = '<memshort-observations>'
// A summary written before the rename still counts as one.
const LEGACY_OPEN = '<aimem-observations>'
// The log row is one dim line until the user presses Show; one switch for every such row in the session.
const showLog = atom({ plugin: 'memshort', key: 'showLog' } as const, false)
const compactRound = atom({ plugin: 'memshort', key: 'compactRound' } as const, 0)
// Tool inputs and results are cut to this many characters for the observer, as pi's serializer does.
const CUT_CHARS = 2000
// After a failed compaction of our own, wait this long before the next try.
const RETRY_MS = 120_000
// The smallest compaction window the engine accepts.
const WINDOW_MIN = 100_000

type ModelUsage = { input_tokens: number; cache_read_input_tokens: number; cache_creation_input_tokens: number }
type Om = { log: string; task: string; summary: string }
// What the log covers: messages before `through`, in the conversation that starts with `base`.
type Checkpoint = { base: string; through: number; mark: string; log: string; task: string }
type View = { start: number; through: number; log: string; task: string }
// One conversation: the main one or a subagent's. `ratio` is real tokens per estimated one, measured from the
// input totals the API reports for this loop's steps against `anchor`.
type Loop = {
  agentId?: string
  checkpoint?: Checkpoint
  observing?: Promise<void>
  wanted: 'none' | 'buffer' | 'compact'
  ratio: number
  phase: 'idle' | 'observing' | 'compacting'
  raw: number
  log: string
  anchor?: { tokens: number; estimated: number }
}

let enabled = false
let settings = { bin: 'memshort', activate: 100_000, midturn: 200_000, keep: 15_000, buffer: 10_000 }
let round = 0
let running = false // a main turn has stepped and not yet completed
let compacting = false
let failedAt = 0
// The engine's window before the mod moved it and the reserve it keeps below it; null when auto-compact is off.
let engine: { limit: number; reserve: number } | null | undefined
let windowSet: number | undefined

// Keyed by agent id, '' for the main conversation. A subagent's entry goes when its turn completes.
const loops = new Map<string, Loop>()
function loopOf(agentId: string | undefined): Loop {
  const key = agentId ?? ''
  let loop = loops.get(key)
  if (!loop) {
    // A new subagent starts from the main conversation's ratio until its own is measured.
    loop = { agentId, wanted: 'none', ratio: loops.get('')?.ratio ?? 1, phase: 'idle', raw: 0, log: '' }
    loops.set(key, loop)
  }
  return loop
}

async function messagesOf($: EngineInterface, loop: Loop): Promise<readonly SessionMessage[]> {
  if (loop.agentId === undefined) return $.session.messages()
  // @ts-expect-error The engine's subagent projection is not yet declared.
  const found = await $.session.messages({ agentId: loop.agentId }) as unknown as SessionMessage[] | { deny: string }
  return 'deny' in found ? [] : found
}

const fingerprint = (m: SessionMessage) =>
  [m.role, m.text.slice(0, 200), ...m.toolUses.map(t => t.tool_use_id), ...(m.toolResults ?? []).map(t => t.tool_use_id)].join('|')

// Characters / 4 of what the transcript shows. A tool's result is counted on the assistant message that called it,
// not again on the user message carrying it. Alone this reads about half the real count: thinking is kept
// encrypted, out of sight, and Claude's tokenizer packs commands, paths and JSON tighter (one measured
// session: 29.7k estimated against 57.3k real).
const estimate = (messages: readonly SessionMessage[]) =>
  Math.ceil(
    messages.reduce(
      (n, m) => n + m.text.length + m.toolUses.reduce((k, t) => k + JSON.stringify(t.input).length + (t.text?.length ?? 0), 0),
      0,
    ) / 4,
  )

// The estimated growth since the anchor before the ratio is trusted.
const CALIBRATE_MIN = 2_000

const size = (loop: Loop, messages: readonly SessionMessage[]) => Math.ceil(estimate(messages) * loop.ratio)

// A step's input total against the estimate of the messages it was sent. A drop (a compaction, a resume that left
// the thinking out) starts the measurement over and keeps the last ratio meanwhile.
function calibrate(loop: Loop, estimated: number, usage: ModelUsage) {
  const tokens = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
  const anchor = loop.anchor
  if (!anchor || tokens < anchor.tokens || estimated < anchor.estimated) {
    loop.anchor = { tokens, estimated }
    return
  }
  if (estimated - anchor.estimated >= CALIBRATE_MIN) loop.ratio = (tokens - anchor.tokens) / (estimated - anchor.estimated)
}

// A tool result stays with the call it answers, so a kept tail never starts with one.
const canCut = (m: SessionMessage) => m.role === 'assistant' || !m.toolResults?.length

function cutIndex(loop: Loop, messages: readonly SessionMessage[], from: number, keep: number): number {
  let kept = 0
  for (let i = messages.length - 1; i > from; i--) {
    const m = messages[i]!
    kept += size(loop, [m])
    if (kept >= keep && canCut(m)) return i
  }
  return from
}

const clip = (text: string) => (text.length > CUT_CHARS ? `${text.slice(0, CUT_CHARS)} [${text.length - CUT_CHARS} more chars]` : text)
const REMINDER = /<system-reminder>[\s\S]*?<\/system-reminder>\s*/g

function transcript(messages: readonly SessionMessage[]): string {
  const out: string[] = []
  for (const m of messages) {
    const text = m.text.replace(REMINDER, '').trim()
    if (m.role === 'user') {
      if (text) out.push(`[User]: ${text}`)
      continue
    }
    if (text) out.push(`[Assistant]: ${text}`)
    if (m.toolUses.length) out.push(`[Assistant tool calls]: ${m.toolUses.map(t => `${t.tool}(${clip(JSON.stringify(t.input))})`).join('; ')}`)
    for (const t of m.toolUses) if (t.text) out.push(`[Tool result]: ${clip(t.text)}`)
  }
  return out.join('\n\n')
}

const SKILL_MESSAGE = 'Base directory for this skill:'

// The slice as ordered transcript parts for `memshort om`: conversation runs, and between them the text kept verbatim
// in the log. That is the first user message when the slice starts the session (`first`), the message Claude Code
// injects when a skill loads ("Base directory for this skill: ..."), and the result of a successful `Read` of a
// SKILL.md. A kept text is not also sent as conversation.
function transcriptParts(messages: readonly SessionMessage[], first: boolean): Part[] {
  const parts = new Parts<SessionMessage>(transcript)
  for (const m of messages) {
    const text = m.text.replace(REMINDER, '').trim()
    if (m.role === 'user') {
      if (text.startsWith(SKILL_MESSAGE)) {
        parts.keep({ kind: 'skill', name: /^Base directory for this skill: (.*)$/m.exec(text)?.[1]?.trim(), content: text })
        first = false
        continue
      }
      if (first && text && !m.toolResults?.length) {
        first = false
        parts.keep({ kind: 'user', content: text })
        continue
      }
      parts.text(m)
      continue
    }
    const reads = m.toolUses.filter(t => {
      const input = t.input as { file_path?: unknown; path?: unknown } | undefined
      const path = input?.file_path ?? input?.path
      return t.tool === 'Read' && typeof path === 'string' && SKILL_FILE.test(path) && t.text
    })
    if (!reads.length) {
      parts.text(m)
      continue
    }
    // The call stays in the conversation run; each skill file's text is kept verbatim after it.
    parts.text({ ...m, toolUses: m.toolUses.map(t => (reads.includes(t) ? { ...t, text: '(skill file, kept verbatim in the log)' } : t)) })
    for (const t of reads) {
      const input = t.input as { file_path?: unknown; path?: unknown }
      parts.keep({ kind: 'skill', name: String(input.file_path ?? input.path), content: t.text! })
    }
  }
  return parts.done()
}

function parseSummary(text: string): { log: string; task: string } | undefined {
  if (!text.includes(OPEN) && !text.includes(LEGACY_OPEN)) return undefined
  const log = logOf(text) ?? ''
  // The task follows the log; a task tag quoted inside a preserved skill cannot stand in for it.
  const after = text.indexOf('<current-task>\n', text.indexOf('<observation-log>\n') + log.length)
  const task = after < 0 ? '' : (/^<current-task>\n([\s\S]*?)\n<\/current-task>/.exec(text.slice(after))?.[1]?.trim() ?? '')
  return { log, task: task === '(none)' ? '' : task }
}

// Where the raw messages start (after our summary) and what the log already covers.
function view(loop: Loop, messages: readonly SessionMessage[]): View {
  const first = messages[0]
  const summary = first ? parseSummary(first.text) : undefined
  const start = summary ? 1 : 0
  const c = loop.checkpoint
  const mark = c ? messages[c.through - 1] : undefined
  if (c && first && mark && c.base === fingerprint(first) && c.through > start && c.mark === fingerprint(mark)) {
    return { start, through: c.through, log: c.log, task: c.task }
  }
  return { start, through: start, log: summary?.log ?? '', task: summary?.task ?? '' }
}

// One persistent UI line for the main conversation. Subagent work must not overwrite its metrics.
async function status($: EngineInterface, loop: Loop, messages?: readonly SessionMessage[]) {
  if (loop.agentId !== undefined) return
  if (messages) {
    const v = view(loop, messages)
    loop.raw = size(loop, messages.slice(v.start))
    loop.log = v.log
  }
  const count = observationsOf(loop.log).match(/^[*-]\s+[🔴🟡🟢]/gmu)?.length ?? 0
  const fmt = (n: number) => `${(n / 1000).toFixed(1)}k`
  const limit = running ? settings.midturn : settings.activate
  await $.ui.status(`OM r${round} · raw ~${fmt(loop.raw)}/${fmt(limit)} · ${count} obs · log ~${fmt(loop.log.length / 3.5)} · ${loop.phase}`)
}

async function om($: EngineInterface, request: { log: string; task: string; transcript: Part[]; focus?: string }): Promise<Om> {
  const ran = await $.process.run([settings.bin, 'om'], { stdin: JSON.stringify(request), timeoutMs: 600_000 })
  let parsed: (Om & { error?: string }) | undefined
  try {
    parsed = JSON.parse(ran.stdout)
  } catch {}
  if (ran.exitCode !== 0 || !parsed || parsed.error) throw new Error(parsed?.error ?? (ran.stderr.trim().slice(-500) || `exit ${ran.exitCode}`))
  return parsed
}

// Moves the engine's compaction window so its own threshold falls where the main conversation's raw messages pass
// compact_during_turn_tokens: what the request carried besides them (system prompt, tools, the log) plus that, plus
// the reserve. Never above the window the session started with; back up to it while a failed compaction waits out
// RETRY_MS.
async function aim($: EngineInterface, loop: Loop, messages: readonly SessionMessage[], usage: ModelUsage) {
  if (engine === undefined) {
    const b = (await $.session.usage({ breakdown: 'summary' })).context.breakdown
    // The original window is kept beside the moved one, so a reload of the mod does not take the moved one for it.
    const kept = Number(await $.env.get('MEMSHORT_ENGINE_WINDOW'))
    engine = b?.autoCompactThreshold
      ? { limit: kept || b.rawMaxTokens, reserve: b.rawMaxTokens - b.autoCompactThreshold }
      : null
    if (engine && !kept) await $.env.set('MEMSHORT_ENGINE_WINDOW', String(engine.limit))
  }
  if (!engine) return
  const tokens = usage.input_tokens + usage.cache_read_input_tokens + usage.cache_creation_input_tokens
  const fixed = tokens - size(loop, messages.slice(view(loop, messages).start))
  const target = Date.now() - failedAt < RETRY_MS ? engine.limit : fixed + settings.midturn + engine.reserve
  const window = Math.min(engine.limit, Math.max(WINDOW_MIN, Math.ceil(target / 1000) * 1000))
  if (window === windowSet) return
  const r = await $.env.set('CLAUDE_CODE_AUTO_COMPACT_WINDOW', String(window)) as unknown as { deny?: string } | undefined
  if (r?.deny) {
    engine = null
    return $.ui.log(`memshort: cannot move the compaction window, compacting between turns only: ${r.deny}`)
  }
  windowSet = window
}

// Observe what has gone unobserved behind the loop's kept tail, if there is enough of it.
async function buffer($: EngineInterface, loop: Loop) {
  const messages = await messagesOf($, loop)
  const v = view(loop, messages)
  const cut = cutIndex(loop, messages, v.through, settings.keep)
  const pending = messages.slice(v.through, cut)
  const first = messages[0]
  const mark = messages[cut - 1]
  if (!first || !mark || size(loop, pending) < settings.buffer) return
  loop.phase = 'observing'
  await status($, loop, messages)
  try {
    const r = await om($, { log: v.log, task: v.task, transcript: transcriptParts(pending, v.through === 0) })
    loop.checkpoint = { base: fingerprint(first), through: cut, mark: fingerprint(mark), log: r.log, task: r.task }
    loop.log = r.log
  } catch (err) {
    $.ui.log(`memshort: background observation failed: ${(err as Error).message}`)
  } finally {
    loop.phase = 'idle'
    await status($, loop)
  }
}

// Hooks may not keep `$`, so they only leave a request on the loop for the clock started in startCompact.
// 'compact' (after a main turn) checks the size and compacts; 'buffer' (during a turn) observes in the background.
const want = (loop: Loop, what: 'buffer' | 'compact') => {
  if (enabled && loop.wanted !== 'compact') loop.wanted = what
}

async function tick($: EngineInterface) {
  if (!enabled) return
  for (const loop of loops.values()) {
    if (loop.wanted === 'none') continue
    if (loop.agentId === undefined && compacting) continue
    const what = loop.wanted
    loop.wanted = 'none'
    if (what === 'compact' && Date.now() - failedAt > RETRY_MS) {
      const messages = await $.session.messages()
      if (size(loop, messages.slice(view(loop, messages).start)) >= settings.activate) {
        compacting = true
        // `/compact` as if typed, queued until the session is idle. Not $.session.compact(): the engine never runs a
        // plugin's hook on an event the plugin raised itself, so its own session.compact hook would be skipped and
        // the engine's summary would run.
        try {
          await $.command.run({ command: 'compact' })
        } catch (err) {
          failedAt = Date.now()
          $.ui.log(`memshort: could not start a compaction: ${(err as Error).message}`)
        } finally {
          compacting = false
        }
        continue
      }
    }
    if (loop.observing) continue
    loop.observing = buffer($, loop).finally(() => {
      loop.observing = undefined
    })
  }
}

// Reset when the host's selection changes across sessions in the same process.
export function stopCompact() {
  enabled = false
  loops.clear()
}

// Called from the mod's session.start (a plugin has one): turns compaction on for interactive sessions.
// `own` holds the thresholds set in the host's selection; environment variables win over them.
export async function startCompact($: EngineInterface, own: Selected['thresholds'] = {}) {
  const role = (await $.env.get('MEMSHORT_ROLE')) ?? 'interactive'
  if (role !== 'interactive' || (await $.env.get('MEMSHORT_COMPACT')) === 'off') return
  const limits = await thresholds($, own)
  settings = {
    bin: (await $.env.get('MEMSHORT_COMPACT_BIN')) ?? 'memshort',
    activate: limits.compact_after_turn_tokens,
    midturn: limits.compact_during_turn_tokens,
    keep: limits.keep_recent_tokens,
    buffer: limits.observe_every_tokens,
  }
  loops.clear()
  running = false
  engine = undefined
  windowSet = undefined
  failedAt = 0
  compacting = false
  round = await read($, compactRound)
  await status($, loopOf(undefined), await $.session.messages())
  enabled = true
  $.clock.every(2000, () => void tick($))
}

export const registerCompact: Register = on => {
  on('turn.step', async function* ($, e, next) {
    if (!enabled) return yield* next(e)
    const loop = loopOf(e.agentId)
    if (e.agentId === undefined) running = true
    want(loop, 'buffer')
    const messages = await messagesOf($, loop)
    await status($, loop, messages)
    const estimated = estimate(messages)
    const step = yield* next(e)
    if (step?.usage) {
      calibrate(loop, estimated, step.usage)
      if (e.agentId === undefined) await aim($, loop, messages, step.usage)
    }
    await status($, loop, await messagesOf($, loop))
    return step
  })

  on('turn.complete', async ($, e, next) => {
    const result = await next(e)
    if (!enabled) return result
    if (e.agentId === undefined) {
      running = false
      const loop = loopOf(undefined)
      want(loop, 'compact')
      await status($, loop, await messagesOf($, loop))
    } else loops.delete(e.agentId)
    return result
  })

  // The observation log stands in the transcript as a user row. User rows always arrive expanded, so ctrl+o cannot
  // tell the views apart: the row is one dim line with a button that shows the whole log.
  on('ui.render', { component: 'UserMessage' }, async ($, e, next) => {
    const summary = parseSummary(e.props.text)
    if (!summary) return next(e)
    const count = observationsOf(summary.log).match(/^\* [🔴🟡🟢]/gmu)?.length ?? 0
    const isShown = await read($, showLog)
    const { Box, Button, Text } = $.ui.resolve(e)
    return (
      <Box flexDirection="column">
        <Box>
          <Text dimColor>memshort: older messages replaced by {count} observations </Text>
          <Button key="memshort-log" label={isShown ? 'Hide' : 'Show'} onPress={() => update($, showLog, (shown: boolean) => !shown)} />
        </Box>
        {isShown ? <Text dimColor>{e.props.text}</Text> : null}
      </Box>
    )
  })

  on('session.compact', async ($, e, next) => {
    // Sessions without memshort keep the engine's own compaction.
    if (!enabled) return next(e)
    // The engine's summary computed ahead of time would never be used.
    if (e.trigger === 'precompute') return { skip: 'memshort compacts with its observation log' }
    const loop = loopOf(e.agentId)
    try {
      await loop.observing
      const v = view(loop, e.messages)
      const cut = cutIndex(loop, e.messages, v.start, settings.keep)
      if (cut <= v.start) return { skip: 'memshort: nothing older than the newest messages to compact' }
      const pending = e.messages.slice(Math.min(v.through, cut), cut)
      loop.phase = 'compacting'
      await status($, loop, e.messages)
      const first = Math.min(v.through, cut) === 0
      const r = await om($, { log: v.log, task: v.task, transcript: transcriptParts(pending, first), focus: e.instructions })
      const summary = r.summary
      loop.checkpoint = undefined // the log now lives in the summary message
      loop.log = r.log
      loop.raw = size(loop, e.messages.slice(cut))
      if (e.agentId === undefined) {
        round++
        await update($, compactRound, () => round)
      }
      return { messages: [{ role: 'user', text: summary, toolUses: [] }, ...e.messages.slice(cut)] }
    } catch (err) {
      // An error, never the engine's own summary in its place: the conversation stays as it is and the skip line
      // says why. The mod's own trigger waits RETRY_MS before trying again.
      failedAt = Date.now()
      // The engine would try again before every request: its window goes back up until RETRY_MS has passed.
      if (engine && windowSet !== engine.limit) {
        await $.env.set('CLAUDE_CODE_AUTO_COMPACT_WINDOW', String(engine.limit))
        windowSet = engine.limit
      }
      return { skip: `memshort compaction failed: ${(err as Error).message}` }
    } finally {
      loop.phase = 'idle'
      await status($, loop)
    }
  }).catch(($, e, next) => {
    if (!enabled || next.called) return next(e)
    failedAt = Date.now()
    return { skip: 'memshort compaction failed in its hook; the conversation was kept unchanged' }
  })
}
