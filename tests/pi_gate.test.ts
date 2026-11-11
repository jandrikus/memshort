import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { featureGate, loadInstallation, thresholds } from '../src/memshort/integrations/pi/lib/installation.ts'

test('shared extensions run only on a host with an explicit valid pi selection', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memshort-gate-'))
  const path = join(dir, 'install.json')
  const env = { MEMSHORT_COMPACT_INSTALL_CONFIG: path }
  try {
    expect(featureGate('compact', 'pi', env)).toEqual({ enabled: false, reason: 'not-selected' })
    writeFileSync(path, JSON.stringify({ features: ['compact'], harnesses: ['claude'] }))
    expect(featureGate('compact', 'pi', env)).toEqual({ enabled: false, reason: 'not-selected' })
    writeFileSync(path, JSON.stringify({ features: ['compact'], harnesses: ['pi'] }))
    expect(featureGate('compact', 'pi', env).enabled).toBe(true)
    writeFileSync(path, '{bad')
    expect(featureGate('compact', 'pi', env)).toMatchObject({ enabled: false, reason: 'invalid' })
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('thresholds come from the environment, then the file, then the defaults, and a bad one is an error', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memshort-thresholds-'))
  const path = join(dir, 'install.json')
  const env = { MEMSHORT_COMPACT_INSTALL_CONFIG: path }
  try {
    writeFileSync(path, JSON.stringify({ features: ['compact'], harnesses: ['pi'], thresholds: { compact_during_turn_tokens: 250000 } }))
    const selected = loadInstallation(env)
    expect(thresholds(selected, {})).toEqual({
      compact_after_turn_tokens: 100000,
      compact_during_turn_tokens: 250000,
      keep_recent_tokens: 15000,
      observe_every_tokens: 10000,
      condense_log_above_tokens: 60000,
    })
    expect(thresholds(selected, { MEMSHORT_COMPACT_DURING_TURN_TOKENS: '300_000' }).compact_during_turn_tokens).toBe(300000)
    expect(() => thresholds(selected, { MEMSHORT_KEEP_RECENT_TOKENS: '1.5k' })).toThrow('MEMSHORT_KEEP_RECENT_TOKENS must be a positive')
    writeFileSync(path, JSON.stringify({ features: ['compact'], harnesses: ['pi'], thresholds: { midturn: 1 } }))
    expect(featureGate('compact', 'pi', env)).toMatchObject({ enabled: false, reason: 'invalid' })
    expect(() => loadInstallation(env)).toThrow('unknown thresholds: midturn')
    writeFileSync(path, JSON.stringify({ features: ['compact'], harnesses: ['pi'], thresholds: { keep_recent_tokens: 0 } }))
    expect(() => loadInstallation(env)).toThrow('keep_recent_tokens must be a positive')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})

test('stage model keys are accepted and checked, not run', () => {
  const dir = mkdtempSync(join(tmpdir(), 'memshort-stages-'))
  const path = join(dir, 'install.json')
  const env = { MEMSHORT_COMPACT_INSTALL_CONFIG: path }
  try {
    writeFileSync(path, JSON.stringify({ features: ['compact'], harnesses: ['pi'], observer_model: 'a/b', reflector_model: 'llamacpp/local-model' }))
    expect(featureGate('compact', 'pi', env)).toMatchObject({ enabled: true })
    writeFileSync(path, JSON.stringify({ features: ['compact'], harnesses: ['pi'], reflector_model: 'nope' }))
    expect(() => loadInstallation(env)).toThrow('reflector_model: model must look like')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
