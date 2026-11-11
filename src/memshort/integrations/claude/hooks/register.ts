import type { Register } from 'claude-code'
import { registerCompact, startCompact, stopCompact } from './compact.tsx'
import { installed } from './installation.ts'

export const register: Register = (on, options) => {
  registerCompact(on, options)
  on('session.start', async ($, e, next) => {
    stopCompact()
    try {
      const selected = await installed($)
      if (selected.features.has('compact')) await startCompact($, selected.thresholds)
    } catch (err) {
      $.ui.log(`memshort: compaction is off: ${(err as Error).message}`)
    }
    return next(e)
  })
}
