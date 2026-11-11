// Shared by the Claude and pi adapters. `memshort om` keeps verbatim text (the first user message, loaded skills)
// inside the observation log as length-prefixed blocks, at the point it appeared (om.py, `split_log`). The adapters
// only send it as transcript parts and read the log back out of a summary.
export type Preserved = { kind: 'user' | 'skill'; content: string; name?: string }
export type Part = { text: string } | { preserved: Preserved }

const BLOCK_OPEN = '<memshort-preserved>'
const BLOCK_CLOSE = '</memshort-preserved>'
const LOG_OPEN = '<observation-log>\n'
const LOG_END = '</observation-log>'

// Advance `count` code points from `pos`; the block length counts code points, as Python's len does.
function skip(text: string, pos: number, count: number): number {
  while (count > 0 && pos < text.length) {
    pos += (text.codePointAt(pos) ?? 0) > 0xffff ? 2 : 1
    count--
  }
  return count === 0 ? pos : -1
}

// The log of a rendered summary, read line by line; each preserved block is stepped over by its length, so a
// closing tag quoted in a skill cannot end the log. Undefined when the summary has no well-formed log.
export function logOf(summary: string): string | undefined {
  let start = summary.indexOf(LOG_OPEN)
  if (start < 0) return undefined
  start += LOG_OPEN.length
  let pos = start
  while (pos < summary.length) {
    if (summary.startsWith(BLOCK_OPEN, pos)) {
      const end = summary.indexOf('\n', pos)
      if (end < 0) return undefined
      let length: unknown
      try {
        length = (JSON.parse(summary.slice(pos + BLOCK_OPEN.length, end)) as { length?: unknown }).length
      } catch {
        return undefined
      }
      if (!Number.isSafeInteger(length) || (length as number) < 0) return undefined
      pos = skip(summary, end + 1, length as number)
      if (pos < 0 || summary[pos] !== '\n' || !summary.startsWith(BLOCK_CLOSE, pos + 1)) return undefined
      pos += 1 + BLOCK_CLOSE.length
    } else {
      let lineEnd = summary.indexOf('\n', pos)
      if (lineEnd < 0) lineEnd = summary.length
      if (summary.slice(pos, lineEnd) === LOG_END) {
        const log = summary.slice(start, Math.max(start, pos - 1))
        return log.trim() === '(empty)' ? '' : log.replace(/^\n+|\n+$/g, '')
      }
      pos = lineEnd
    }
    if (summary[pos] !== '\n') return undefined
    pos++
  }
  return undefined
}

// The model-written lines of a log, for counting observations: the preserved blocks' text is left out.
export function observationsOf(log: string): string {
  const out: string[] = []
  let pos = 0
  for (;;) {
    const at = log.indexOf(BLOCK_OPEN, pos)
    if (at < 0 || (at > 0 && log[at - 1] !== '\n')) {
      if (at < 0) break
      out.push(log.slice(pos, at + 1))
      pos = at + 1
      continue
    }
    out.push(log.slice(pos, at))
    const end = log.indexOf('\n', at)
    let length: unknown
    try {
      length = (JSON.parse(log.slice(at + BLOCK_OPEN.length, end)) as { length?: unknown }).length
    } catch {
      return out.join('')
    }
    const close = Number.isSafeInteger(length) ? skip(log, end + 1, length as number) : -1
    if (close < 0) return out.join('')
    pos = close + 1 + BLOCK_CLOSE.length
  }
  out.push(log.slice(pos))
  return out.join('')
}

// Builds the ordered parts: text runs are collected and serialized together, a preserved item ends the run.
export class Parts<T> {
  private run: T[] = []
  readonly parts: Part[] = []
  constructor(private readonly serialize: (items: T[]) => string) {}
  text(item: T) {
    this.run.push(item)
  }
  keep(item: Preserved) {
    this.flush()
    if (item.content.trim()) this.parts.push({ preserved: item })
  }
  done(): Part[] {
    this.flush()
    return this.parts
  }
  private flush() {
    if (!this.run.length) return
    const text = this.serialize(this.run)
    this.run = []
    if (text.trim()) this.parts.push({ text })
  }
}

export const SKILL_FILE = /(?:^|\/)SKILL\.md$/
