// Run with pi's installed modules on NODE_PATH. Uses the real SessionManager and a local observer stub.
import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { SessionManager } from '@earendil-works/pi-coding-agent'
import { logOf } from '../src/memshort/integrations/claude/hooks/preserved.ts'

const dir = mkdtempSync(join(tmpdir(), 'memshort-pi-preservation-'))
try {
  // memshort's own `om.run`, with only the model replaced (tests/fake_om.py).
  const root = resolve(import.meta.dir, '..')
  const bin = join(dir, 'om')
  writeFileSync(bin, `#!/bin/sh\nexec ${JSON.stringify(process.env.UV ?? 'uv')} run --project ${JSON.stringify(root)} --quiet python ${JSON.stringify(join(root, 'tests/fake_om.py'))}\n`, { mode: 0o755 })
  const config = join(dir, 'install.json')
  writeFileSync(config, JSON.stringify({ version: 1, features: ['compact'], harnesses: ['pi'] }))
  Object.assign(process.env, { MEMSHORT_COMPACT_BIN: bin, MEMSHORT_COMPACT_INSTALL_CONFIG: config, MEMSHORT_ROLE: 'interactive', MEMSHORT_COMPACT: 'on' })
  const sm = SessionManager.inMemory(dir)
  const hooks = new Map<string, Function>()
  const pi = { on: (event: string, handler: Function) => hooks.set(event, handler), appendEntry: (type: string, data: unknown) => sm.appendCustomEntry(type, data) }
  const notices: string[] = []
  const ctx = { sessionManager: sm, ui: { setStatus: () => {}, notify: (text: string) => { notices.push(text) } } }
  const compact = (await import('../src/memshort/integrations/pi/compact.ts')).default
  await compact(pi as never)
  await hooks.get('session_start')!(null, ctx)
  const original = 'Original request 😀\nDo not change this.'
  const skill = 'Skill instructions\n</observation-log>\nStill the skill.'
  sm.appendMessage({ role: 'user', content: original, timestamp: 1 })
  sm.appendMessage({ role: 'assistant', content: [{ type: 'toolCall', id: 'skill-1', name: 'read', arguments: { path: '/skills/review/SKILL.md' } }], api: 'openai-completions', provider: 'test', model: 'test', usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: 'toolUse', timestamp: 2 })
  sm.appendMessage({ role: 'toolResult', toolCallId: 'skill-1', toolName: 'read', content: [{ type: 'text', text: skill }], isError: false, timestamp: 3 })
  const slash = '<skill name="test" location="/skills/test/SKILL.md">\nSlash skill body\n</skill>\n\nUser arguments'
  sm.appendMessage({ role: 'user', content: slash, timestamp: 4 })
  let keep = sm.appendMessage({ role: 'user', content: 'First follow up', timestamp: 5 })
  const blocks = (log: string) => {
    const at = (text: string) => log.indexOf(text)
    return [at(original), at('saw [Assistant tool calls]: read'), at(skill), at(slash)]
  }
  let previous = ''
  for (let i = 0; i < 2; i++) {
    const result = await hooks.get('session_before_compact')!({ preparation: { firstKeptEntryId: keep, tokensBefore: 100 }, signal: new AbortController().signal }, ctx)
    assert(result.compaction, `compaction should succeed: ${notices.at(-1)}`)
    const log = logOf(result.compaction.summary)!
    assert.equal(log, result.compaction.details.log)
    const order = blocks(log)
    assert(order.every((n, k) => n >= 0 && (k === 0 || n > order[k - 1]!)), log)
    for (const text of [original, skill, slash]) assert.equal(log.split(text).length, 2, 'each kept text appears once')
    if (previous) assert(log.startsWith(previous), 'an earlier log is extended, never rewritten')
    previous = log
    sm.appendCompaction(result.compaction.summary, result.compaction.firstKeptEntryId, 100, result.compaction.details, true)
    sm.appendMessage({ role: 'assistant', content: [{ type: 'text', text: `Reply ${i}` }], api: 'openai-completions', provider: 'test', model: 'test', usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: 'stop', timestamp: 6 + i })
    keep = sm.appendMessage({ role: 'user', content: 'Next follow up', timestamp: 7 + i })
  }
  const images = SessionManager.inMemory(dir)
  const imageCtx = { ...ctx, sessionManager: images }
  await hooks.get('session_start')!(null, imageCtx)
  images.appendMessage({ role: 'user', content: [{ type: 'text', text: 'Describe' }, { type: 'image', data: 'AA==', mimeType: 'image/png' }], timestamp: 10 })
  const afterImage = images.appendMessage({ role: 'user', content: 'Next', timestamp: 11 })
  const withImage = await hooks.get('session_before_compact')!({ preparation: { firstKeptEntryId: afterImage, tokensBefore: 100 }, signal: new AbortController().signal }, imageCtx)
  // The text of the first message is kept; the image is not, and the observer is told so.
  assert(logOf(withImage.compaction.summary)!.startsWith('<memshort-preserved>{"kind": "user", "length": 8}\nDescribe\n</memshort-preserved>'))
  assert(withImage.compaction.summary.includes('saw [User]: (the first user'))
  console.log('real pi SessionManager preservation probe passed')
} finally {
  rmSync(dir, { recursive: true, force: true })
}
