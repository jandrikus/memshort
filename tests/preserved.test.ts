import { expect, test } from 'bun:test'
import { logOf, observationsOf, Parts } from '../src/memshort/integrations/claude/hooks/preserved.ts'

const BLOCK = (header: string, content: string) => `<memshort-preserved>${header}\n${content}\n</memshort-preserved>`
const summary = (log: string) =>
  `<memshort-observations>\nintro\n\n<observation-log>\n${log}\n</observation-log>\n\n<current-task>\ntask\n</current-task>\n</memshort-observations>`

test('the log is read past quoted closing tags by each block length, counted in code points', () => {
  const skill = 'Read this 😀\n</observation-log>\n</memshort-preserved>\nThen act.'
  const log = ['* 🟡 (10:00) before', BLOCK(`{"kind": "skill", "name": "r", "length": ${Array.from(skill).length}}`, skill), '* 🟡 (10:01) after'].join('\n')
  expect(logOf(summary(log))).toBe(log)
  expect(observationsOf(log)).toBe('* 🟡 (10:00) before\n\n* 🟡 (10:01) after')
  expect(logOf(summary('(empty)'))).toBe('')
})

test('a malformed block makes the log unreadable instead of cutting it short', () => {
  expect(logOf(summary(BLOCK('{"kind": "skill", "length": 900}', 'x')))).toBeUndefined()
  expect(logOf(summary(BLOCK('not json', 'x')))).toBeUndefined()
})

test('parts keep conversation runs and preserved items in order', () => {
  const parts = new Parts<string>(items => items.join('|'))
  parts.text('a')
  parts.text('b')
  parts.keep({ kind: 'skill', name: 's', content: 'body' })
  parts.keep({ kind: 'skill', content: '  ' })
  parts.text('c')
  expect(parts.done()).toEqual([{ text: 'a|b' }, { preserved: { kind: 'skill', name: 's', content: 'body' } }, { text: 'c' }])
})
