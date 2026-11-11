// The mod's $.state contract.
declare module 'claude-code' {
  interface PluginState {
    // Whether the transcript shows the observation log a compaction put in place of older messages.
    'memshort': { showLog: boolean; compactRound: number }
  }
}
