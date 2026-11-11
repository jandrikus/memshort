import { expect, mock, test } from 'bun:test'

mock.module('react/jsx-dev-runtime', () => ({ jsxDEV: () => null }))
mock.module('react/jsx-runtime', () => ({ jsx: () => null, jsxs: () => null }))
mock.module('claude-code', () => ({
  atom: (_key: unknown, value: unknown) => ({ value }),
  read: async (_$: unknown, atom: { value: unknown }) => atom.value,
  update: (_$: unknown, atom: { value: unknown }, change: (value: unknown) => unknown) => { atom.value = change(atom.value) },
}))
const { register } = await import('../src/memshort/integrations/claude/hooks/register')

test('compact-only Claude registration never invokes memory tools, retrieval, or host writes', async () => {
  const hooks = new Map<string, Function>()
  const on = (name: string, ...args: unknown[]) => {
    hooks.set(name, args.at(-1) as Function)
    return { catch: () => {} }
  }
  register(on as never, {} as never)
  let clocks = 0
  const calls: string[] = []
  const $ = {
    env: { get: async (key: string) => key === 'HOME' ? '/home/test' : undefined },
    fs: {
      exists: async (path: string) => { calls.push(`exists:${path}`); return true },
      read: async () => JSON.stringify({ features: ['compact'], harnesses: ['claude'] }),
      write: async () => { throw new Error('unexpected host tracking') },
    },
    clock: { every: () => { clocks++ } },
    process: { run: async () => { throw new Error('unexpected memory subprocess') } },
    tool: { register: async () => { throw new Error('unexpected memory tool') } },
    session: { messages: async () => [] },
    ui: { log: () => {}, status: (text: string) => { calls.push(`status:${text}`) } },
  }
  await hooks.get('session.start')!($, { isInteractive: true }, async (e: unknown) => e)
  expect(clocks).toBe(1)
  expect(calls).toEqual([
    'exists:/home/test/.config/memshort/install.json',
    'status:OM r0 · raw ~0.0k/100.0k · 0 obs · log ~0.0k · idle',
  ])
  expect(hooks.has('prompt.submit')).toBe(false)
  expect(hooks.has('tool.call')).toBe(false)
})

test('a host without a selection does not run compaction, including after a prior enabled session', async () => {
  const hooks = new Map<string, Function>()
  register(((name: string, ...args: unknown[]) => {
    hooks.set(name, args.at(-1) as Function)
    return { catch: () => {} }
  }) as never, {} as never)
  let selected = true
  const $ = {
    env: { get: async (key: string) => key === 'HOME' ? '/tmp/home' : undefined },
    fs: { exists: async () => selected, read: async () => JSON.stringify({ features: ['compact'], harnesses: ['claude'] }) },
    clock: { every: () => {} },
    session: { messages: async () => [] },
    ui: { status: () => {}, log: () => {} },
  }
  await hooks.get('session.start')!($, {}, async (e: unknown) => e)
  selected = false
  await hooks.get('session.start')!($, {}, async (e: unknown) => e)
  const event = { messages: [], trigger: 'manual' }
  expect(await hooks.get('session.compact')!($, event, async () => 'native')).toBe('native')
  expect(await hooks.get('turn.complete')!($, {}, async () => 'done')).toBe('done')
})

test('Claude uses separate turn and midturn thresholds', async () => {
  const hooks = new Map<string, Function>()
  register(((name: string, ...args: unknown[]) => {
    hooks.set(name, args.at(-1) as Function)
    return { catch: () => {} }
  }) as never, {} as never)
  const messages = [{ role: 'user', text: 'x'.repeat(400_000), toolUses: [] }]
  const writes: [string, string][] = []
  let tick: () => Promise<void> = async () => {}
  let queued = 0
  const $ = {
    env: {
      get: async () => undefined,
      set: async (key: string, value: string) => { writes.push([key, value]) },
    },
    clock: { every: (_ms: number, fn: () => Promise<void>) => { tick = fn } },
    command: { run: async () => { queued++ } },
    session: {
      messages: async () => messages,
      usage: async () => ({ context: { breakdown: { autoCompactThreshold: 467_000, rawMaxTokens: 500_000 } } }),
    },
    ui: { status: () => {}, log: () => {} },
  }
  const { startCompact } = await import('../src/memshort/integrations/claude/hooks/compact')
  await startCompact($ as never)
  const next = async function* () {
    return { usage: { input_tokens: 150_000, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 } }
  }
  for await (const _ of hooks.get('turn.step')!($, {}, next)) { /* drain */ }
  expect(writes).toContainEqual(['CLAUDE_CODE_AUTO_COMPACT_WINDOW', '283000'])
  await hooks.get('turn.complete')!($, {}, async () => ({}))
  await tick()
  expect(queued).toBe(1)
})

test('Claude status tracks compactions and carries the previous log without putting metrics in the summary', async () => {
  const hooks = new Map<string, Function>()
  register(((name: string, ...args: unknown[]) => {
    hooks.set(name, args.at(-1) as Function)
    return { catch: () => {} }
  }) as never, {} as never)
  const statuses: string[] = []
  const requests: { log: string; transcript: string }[] = []
  let messages = ['user', 'assistant', 'user'].map(role => ({ role, text: 'text '.repeat(40), toolUses: [] }))
  let fail = false
  const $ = {
    env: { get: async (key: string) => key === 'MEMSHORT_KEEP_RECENT_TOKENS' ? '10' : undefined },
    clock: { every: () => {} },
    session: { messages: async () => messages },
    ui: { status: (text: string) => { statuses.push(text) }, log: () => {} },
    process: { run: async (_args: unknown, opts: { stdin: string }) => {
      const request = JSON.parse(opts.stdin)
      requests.push(request)
      if (fail) return { exitCode: 1, stdout: '', stderr: 'offline' }
      const log = [request.log, '* 🟡 saved fact'].filter(Boolean).join('\n')
      const summary = `<memshort-observations>\n<observation-log>\n${log}\n</observation-log>\n<current-task>\ntask\n</current-task>\n</memshort-observations>`
      return { exitCode: 0, stdout: JSON.stringify({ log, task: 'task', summary }), stderr: '' }
    } },
  }
  const { startCompact } = await import('../src/memshort/integrations/claude/hooks/compact')
  await startCompact($ as never)
  expect(statuses.at(-1)).toContain('OM r0')
  for (let round = 1; round <= 2; round++) {
    const result = await hooks.get('session.compact')!($, { messages, trigger: 'manual' }, () => { throw new Error('native summary') })
    messages = result.messages
    expect(statuses.at(-1)).toContain(`OM r${round}`)
    expect(statuses.at(-1)).toContain(`${round} obs`)
    expect(statuses.at(-1)).toEndWith('idle')
    expect(messages[0]!.text).not.toContain('OM r')
    expect(statuses.some(text => text.endsWith('compacting'))).toBe(true)
    messages.push({ role: 'user', text: 'more '.repeat(40), toolUses: [] })
  }
  expect(requests[1]!.log).toBe('* 🟡 saved fact')
  fail = true
  const result = await hooks.get('session.compact')!($, { messages, trigger: 'manual' }, () => {})
  expect(result.skip).toContain('offline')
  expect(statuses.at(-1)).toContain('OM r2')
  await startCompact($ as never)
  expect(statuses.at(-1)).toContain('OM r2') // the round survives an adapter reload
})

test('Claude keeps the first user message and loaded skills in conversation order across compactions', async () => {
  const hooks = new Map<string, Function>()
  register(((name: string, ...args: unknown[]) => {
    hooks.set(name, args.at(-1) as Function)
    return { catch: () => {} }
  }) as never, {} as never)
  const skill = 'Base directory for this skill: /skills/review\n\nReview every change.\n</observation-log>\n😀'
  const file = 'File skill\n</memshort-preserved>\nstill the file'
  let messages = [
    { role: 'user', text: 'Original request', toolUses: [] },
    { role: 'assistant', text: 'Looking around', toolUses: [{ tool: 'Skill', input: { skill: 'review' }, tool_use_id: 's1' }] },
    { role: 'user', text: skill, toolUses: [] },
    { role: 'assistant', text: 'Reviewed', toolUses: [{ tool: 'Read', input: { file_path: '/skills/fix/SKILL.md' }, tool_use_id: 'r1', text: file }] },
    { role: 'assistant', text: 'Fixed', toolUses: [] },
    { role: 'user', text: 'Follow up '.repeat(50), toolUses: [] },
  ]
  // memshort's own `om.run` with only the model replaced (tests/fake_om.py).
  const root = new URL('..', import.meta.url).pathname
  const run = async (_cmd: string[], options: { stdin: string }) => {
    const child = Bun.spawn([process.env.UV ?? 'uv', 'run', '--project', root, '--quiet', 'python', `${root}tests/fake_om.py`], { stdin: new Blob([options.stdin]), stdout: 'pipe', stderr: 'pipe' })
    const [stdout, stderr, exitCode] = [await new Response(child.stdout).text(), await new Response(child.stderr).text(), await child.exited]
    return { exitCode, stdout, stderr }
  }
  const $ = {
    env: { get: async (key: string) => key === 'MEMSHORT_KEEP_RECENT_TOKENS' ? '10' : undefined },
    clock: { every: () => {} },
    session: { messages: async () => messages },
    ui: { status: () => {}, log: () => {} },
    process: { run },
  }
  const { startCompact } = await import('../src/memshort/integrations/claude/hooks/compact')
  const { logOf } = await import('../src/memshort/integrations/claude/hooks/preserved')
  await startCompact($ as never)
  let previous = ''
  for (let i = 0; i < 2; i++) {
    const result = await hooks.get('session.compact')!($, { messages, trigger: 'manual' }, () => {})
    expect(result.skip).toBeUndefined()
    messages = result.messages
    const log = logOf(messages[0]!.text)!
    const order = ['Original request', 'saw [Assistant]: Looking around', skill, 'saw [Assistant]: Reviewed', file, 'saw [Assistant]: Fixed'].map(t => log.indexOf(t))
    expect(order.every((n, k) => n >= 0 && (k === 0 || n > order[k - 1]!))).toBe(true)
    expect(log.split(skill)).toHaveLength(2)
    expect(log).toContain('{"kind": "skill", "name": "/skills/review", "length"')
    if (previous) expect(log.startsWith(previous)).toBe(true)
    previous = log
    messages.push({ role: 'assistant', text: `Reply ${i} `.repeat(30), toolUses: [] }, { role: 'user', text: 'Next '.repeat(50), toolUses: [] })
  }
}, 30_000)

test('Claude status shows the during-turn limit while a main turn runs, and file thresholds apply', async () => {
  const hooks = new Map<string, Function>()
  register(((name: string, ...args: unknown[]) => {
    hooks.set(name, args.at(-1) as Function)
    return { catch: () => {} }
  }) as never, {} as never)
  const statuses: string[] = []
  const logs: string[] = []
  let file: unknown = { features: ['compact'], harnesses: ['claude'], thresholds: { compact_during_turn_tokens: 250000 } }
  const $ = {
    env: {
      get: async (key: string) => ({ HOME: '/home/test', MEMSHORT_COMPACT_AFTER_TURN_TOKENS: '120_000' } as Record<string, string>)[key],
      set: async () => {},
    },
    fs: { exists: async () => true, read: async () => JSON.stringify(file) },
    clock: { every: () => {} },
    session: { messages: async () => [], usage: async () => ({ context: { breakdown: null } }) },
    ui: { status: (text: string) => { statuses.push(text) }, log: (text: string) => { logs.push(text) } },
  }
  await hooks.get('session.start')!($, {}, async (e: unknown) => e)
  expect(statuses.at(-1)).toContain('raw ~0.0k/120.0k')
  const next = async function* () { return {} }
  for await (const _ of hooks.get('turn.step')!($, {}, next)) { /* drain */ }
  expect(statuses.at(-1)).toContain('raw ~0.0k/250.0k')
  await hooks.get('turn.complete')!($, {}, async () => ({}))
  expect(statuses.at(-1)).toContain('raw ~0.0k/120.0k')
  file = { features: ['compact'], harnesses: ['claude'], thresholds: { midturn: 1 } }
  statuses.length = 0
  await hooks.get('session.start')!($, {}, async (e: unknown) => e)
  expect(statuses).toEqual([])
  expect(logs.at(-1)).toContain('unknown threshold midturn')
})

test('Claude accepts stage model keys and refuses a bad one', async () => {
  let file: unknown = { features: ['compact'], harnesses: ['claude'], observer_model: 'a/b', reflector_model: 'c/d' }
  const $ = {
    env: { get: async (key: string) => (key === 'HOME' ? '/home/test' : undefined) },
    fs: { exists: async () => true, read: async () => JSON.stringify(file) },
  }
  const { installed } = await import('../src/memshort/integrations/claude/hooks/installation.ts')
  expect([...(await installed($ as never)).features]).toEqual(['compact'])
  file = { features: ['compact'], harnesses: ['claude'], reflector_model: 'nope' }
  await expect(installed($ as never)).rejects.toThrow('invalid installation choices')
})
