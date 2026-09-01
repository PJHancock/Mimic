export type PhaseSegment = { phase: string; start: number; end: number }
export type SimTransition = { timestamp_s: number; phase: string }
export type SyncWindow = {
  phase: string
  rawStart: number
  rawEnd: number
  simStart: number
  simEnd: number
}

export function collapseSimPhases(transitions: SimTransition[]) {
  return transitions.filter((transition, index, all) => index === 0 || all[index - 1].phase !== transition.phase)
}

function precedingIdle(segments: PhaseSegment[], rawIndex: number) {
  for (let index = rawIndex - 1; index >= 0; index -= 1) {
    if (segments[index].phase === 'IDLE') return segments[index]
  }
  return undefined
}

export function syncWindowsFor(
  rawSegments: PhaseSegment[],
  transitions: SimTransition[],
  simEndTime: number,
): SyncWindow[] {
  const simStarts = collapseSimPhases(transitions)
  const windows: SyncWindow[] = []
  let rawIndex = 0
  for (let simIndex = 0; simIndex < simStarts.length; simIndex += 1) {
    const simStart = simStarts[simIndex].timestamp_s
    const simEnd = simStarts[simIndex + 1]?.timestamp_s ?? simEndTime
    if (!(simEnd > simStart)) continue
    const phase = simStarts[simIndex].phase
    while (rawIndex < rawSegments.length && rawSegments[rawIndex].phase === 'IDLE' && phase !== 'IDLE') {
      rawIndex += 1
    }
    const raw = rawSegments[rawIndex]
    if (raw && raw.phase === phase && raw.end >= raw.start) {
      windows.push({ phase, rawStart: raw.start, rawEnd: raw.end, simStart, simEnd })
      rawIndex += 1
      continue
    }
    // Continuation hover is generated in execution even when the classifier
    // stayed in IDLE. Play that intervening footage instead of jumping ahead.
    if (phase !== 'HOVER') continue
    const idle = precedingIdle(rawSegments, rawIndex)
    const holdAt = raw?.start ?? idle?.end ?? 0
    const rawStart = idle && idle.end > idle.start ? idle.start : holdAt
    const rawEnd = idle && idle.end > idle.start ? idle.end : holdAt
    windows.push({ phase, rawStart, rawEnd: Math.max(rawEnd, rawStart), simStart, simEnd })
  }
  return windows
}

export function windowForSimTime(windows: SyncWindow[], simTime: number) {
  if (!windows.length) return undefined
  return windows.find((item) => simTime >= item.simStart && simTime < item.simEnd)
    ?? (simTime < windows[0].simStart ? windows[0] : windows.at(-1))
}
