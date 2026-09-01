import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { syncWindowsFor, windowForSimTime } from './syncWindows'

type Run = { id: string; name: string; completed: boolean; success: boolean | null }
type RawVideo = { name: string; stem: string; size_bytes: number; has_results: boolean }
type RobotConfig = { id: string; name: string; default: boolean }
type PipelineJob = {
  status: 'idle' | 'running' | 'succeeded' | 'failed'
  progress: number; stage: string; message: string; video: string | null; run_id: string | null
  logs: string[]; exit_code: number | null; started_at: string | null; finished_at: string | null
}
type Action = { frame_idx: number; timestamp_s?: number; phase: string; confidence?: number }
type ScoreFrame = { state_scores: Record<string, number> }
type Point = [number, number, number]
type Detail = {
  id: string; name: string
  video: { duration_s?: number; frame_count?: number; fps?: number } | null
  catalog: { labels: string[]; fingerprint?: string } | null
  resolved_actions: Action[]; score_frames: ScoreFrame[]
  waypoints: { path?: { position: Point }[] } | null
  execution: {
    result: { success: boolean; grasp_occurred: boolean; transported: boolean; released: boolean; final_position_error_m: number | null; failure?: string | null; final_state?: { timestamp_s?: number } } | null
    transitions: { timestamp_s: number; phase: string; skill: string; step: number }[]
    samples: { object_position?: Point }[]
  }
  source_video_url: string | null
  simulation_video_url: string | null
}

const cssPhase: Record<string, string> = { IDLE: 'idle', HOVER: 'reach', GRASP: 'grasp', CARRY: 'move', RELEASE: 'release' }
const canonicalPhases = ['IDLE', 'HOVER', 'GRASP', 'CARRY', 'RELEASE']

function time(seconds = 0) {
  return `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, '0')}`
}

function segmentsFor(actions: Action[]) {
  const segments: { phase: string; start: number; end: number }[] = []
  actions.forEach((action, index) => {
    const at = action.timestamp_s ?? index
    const previous = segments.at(-1)
    if (!previous || previous.phase !== action.phase) segments.push({ phase: action.phase, start: at, end: at })
    else previous.end = at
  })
  if (segments.length) segments.at(-1)!.end += actions.length > 1 ? ((actions.at(-1)!.timestamp_s ?? actions.length - 1) / (actions.length - 1)) : .033
  return segments
}

function PanelTitle({ eyebrow, title, end }: { eyebrow: string; title: string; end?: ReactNode }) {
  return <div className="panel-title"><div><span>{eyebrow}</span><h2>{title}</h2></div>{end}</div>
}

function Timeline({ segments, phases, duration, progress, onSeek }: { segments: ReturnType<typeof segmentsFor>; phases: string[]; duration: number; progress: number; onSeek: (value: number) => void }) {
  return <section className="panel timeline-panel">
    <PanelTitle eyebrow="Synchronized inference" title="Skill timeline" end={<span className="mono muted">{time(progress * duration)} / {time(duration)}</span>} />
    <div className="timeline" onClick={(event) => { const box = event.currentTarget.getBoundingClientRect(); onSeek((event.clientX - box.left) / box.width) }}>
      <div className="timeline-track">
        {segments.map((segment, index) => <div className={`segment ${cssPhase[segment.phase] ?? 'idle'}`} key={`${segment.phase}-${index}`} style={{ left: `${segment.start / duration * 100}%`, width: `${Math.max(.5, (segment.end - segment.start) / duration * 100)}%` }}><span>{segment.phase}</span></div>)}
        <div className="playhead" style={{ left: `${progress * 100}%` }}><i /></div>
      </div>
      <div className="ticks">{[0, .25, .5, .75, 1].map((tick) => <span key={tick} style={{ left: `${tick * 100}%` }}>{time(tick * duration)}</span>)}</div>
    </div>
    <div className="legend">{phases.map((phase) => <span key={phase}><i className={cssPhase[phase] ?? 'idle'} />{phase}</span>)}</div>
  </section>
}

function Trajectory({ detail }: { detail: Detail }) {
  const planned = detail.waypoints?.path?.map((waypoint) => waypoint.position) ?? []
  const actual = detail.execution.samples.map((sample) => sample.object_position).filter((point): point is Point => Boolean(point))
  const points = [...planned, ...actual]
  if (!points.length) return <div className="empty">No world trajectory artifact available.</div>
  const xs = points.map((p) => p[0]); const ys = points.map((p) => p[1])
  const minX = Math.min(...xs); const maxX = Math.max(...xs); const minY = Math.min(...ys); const maxY = Math.max(...ys)
  const padX = Math.max(.04, (maxX - minX) * .18); const padY = Math.max(.04, (maxY - minY) * .18)
  const map = (p: Point) => [28 + ((p[1] - minY + padY) / (maxY - minY + padY * 2)) * 464, 218 - ((p[0] - minX + padX) / (maxX - minX + padX * 2)) * 188]
  const line = (series: Point[]) => series.map((point) => map(point).join(',')).join(' ')
  const start = planned[0] && map(planned[0]); const goal = planned.at(-1) && map(planned.at(-1)!)
  return <div className="trajectory-wrap">
    <svg className="trajectory" viewBox="0 0 520 245" role="img" aria-label="Top-down planned and measured world trajectory">
      <defs><pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse"><path d="M32 0H0V32" fill="none" stroke="#232a32" strokeWidth="1" /></pattern></defs>
      <rect width="520" height="245" fill="url(#grid)" /><line x1="28" y1="218" x2="500" y2="218" className="axis" /><line x1="28" y1="18" x2="28" y2="218" className="axis" />
      <text x="470" y="238" className="axis-label">WORLD Y</text><text x="8" y="30" className="axis-label">X</text>
      {actual.length > 1 && <polyline points={line(actual)} className="actual-path" />}{planned.length > 1 && <polyline points={line(planned)} className="planned-path" />}
      {planned.map((point, index) => { const [x, y] = map(point); return <circle key={index} cx={x} cy={y} r="2.3" className="waypoint" /> })}
      {start && <><circle cx={start[0]} cy={start[1]} r="6" className="start-dot" /><text x={start[0] + 10} y={start[1] + 4} className="point-label">GRASP</text></>}
      {goal && <><circle cx={goal[0]} cy={goal[1]} r="7" className="goal-dot" /><text x={goal[0] + 10} y={goal[1] + 4} className="point-label">GOAL</text></>}
    </svg>
    <div className="chart-legend"><span><i className="line planned" />Planned waypoints</span><span><i className="line actual" />Measured object</span><span className="mono">{planned.length} PTS</span></div>
  </div>
}

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]); const [runId, setRunId] = useState('')
  const [detail, setDetail] = useState<Detail | null>(null); const [loading, setLoading] = useState(true); const [error, setError] = useState(''); const [progress, setProgress] = useState(0)
  const [rawVideos, setRawVideos] = useState<RawVideo[]>([]); const [selectedRaw, setSelectedRaw] = useState(''); const [robotConfigs, setRobotConfigs] = useState<RobotConfig[]>([]); const [selectedConfig, setSelectedConfig] = useState(''); const [device, setDevice] = useState('cpu')
  const [pipelineJob, setPipelineJob] = useState<PipelineJob | null>(null); const [pipelineError, setPipelineError] = useState(''); const [artifactRevision, setArtifactRevision] = useState(0)
  const [phaseSync, setPhaseSync] = useState(false); const [syncPlaying, setSyncPlaying] = useState(false); const [syncRate, setSyncRate] = useState(1); const [syncPhase, setSyncPhase] = useState('—'); const [simDuration, setSimDuration] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)
  const simVideoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => { fetch('/api/runs').then((r) => { if (!r.ok) throw new Error('Could not read results/'); return r.json() }).then((data: Run[]) => { setRuns(data); if (data[0]) setRunId(data[0].id); else { setLoading(false); setError('No result artifacts were found.') } }).catch((cause) => { setLoading(false); setError(String(cause)) }) }, [])
  useEffect(() => { Promise.all([fetch('/api/raw-videos').then((r) => r.json()), fetch('/api/robot-configs').then((r) => r.json()), fetch('/api/process').then((r) => r.json())]).then(([videos, configs, job]: [RawVideo[], RobotConfig[], PipelineJob]) => { setRawVideos(videos); setSelectedRaw((current) => current || videos[0]?.name || ''); setRobotConfigs(configs); setSelectedConfig((current) => current || configs.find((item) => item.default)?.id || configs[0]?.id || ''); setPipelineJob(job) }).catch((cause) => setPipelineError(String(cause))) }, [])
  useEffect(() => { if (!runId) return; setLoading(true); setProgress(0); setPhaseSync(false); setSyncPlaying(false); fetch(`/api/runs/${encodeURIComponent(runId)}`).then((r) => { if (!r.ok) throw new Error('Run artifacts could not be loaded.'); return r.json() }).then((data) => { setDetail(data); setLoading(false); setError('') }).catch((cause) => { setLoading(false); setError(String(cause)) }) }, [runId, artifactRevision])
  useEffect(() => {
    if (pipelineJob?.status !== 'running') return
    let finished = false
    const timer = window.setInterval(() => fetch('/api/process').then((r) => r.json()).then((job: PipelineJob) => {
      setPipelineJob(job)
      if (!finished && job.status !== 'running') {
        finished = true; window.clearInterval(timer)
        void fetch('/api/raw-videos').then((r) => r.json()).then(setRawVideos)
        void fetch('/api/runs').then((r) => r.json()).then((updated: Run[]) => { setRuns(updated); if (job.run_id) setRunId(job.run_id); setArtifactRevision((value) => value + 1) })
      }
    }).catch((cause) => setPipelineError(String(cause))), 650)
    return () => window.clearInterval(timer)
  }, [pipelineJob?.status])

  const segments = useMemo(() => segmentsFor(detail?.resolved_actions ?? []), [detail])
  const duration = detail?.video?.duration_s ?? segments.at(-1)?.end ?? 1
  const actionIndex = detail?.resolved_actions.length ? Math.min(detail.resolved_actions.length - 1, Math.floor(progress * detail.resolved_actions.length)) : 0
  const scoreIndex = detail?.score_frames.length ? Math.min(detail.score_frames.length - 1, Math.floor(progress * detail.score_frames.length)) : 0
  const action = detail?.resolved_actions[actionIndex]; const scores = detail?.score_frames[scoreIndex]?.state_scores ?? {}; const confidence = action ? scores[action.phase] ?? action.confidence ?? 0 : 0
  const phases = detail?.catalog?.labels ?? canonicalPhases
  const result = detail?.execution.result
  const syncWindows = useMemo(
    () => syncWindowsFor(
      segments,
      detail?.execution.transitions ?? [],
      result?.final_state?.timestamp_s ?? simDuration,
    ),
    [segments, detail?.execution.transitions, result?.final_state?.timestamp_s, simDuration],
  )
  const syncAvailable = Boolean(detail?.source_video_url && detail?.simulation_video_url && syncWindows.length)
  const synchronizeAt = (simTime: number) => {
    const rawVideo = videoRef.current
    const phaseWindow = windowForSimTime(syncWindows, simTime)
    if (!rawVideo || !phaseWindow) return
    const simSpan = Math.max(1e-6, phaseWindow.simEnd - phaseWindow.simStart)
    const rawSpan = Math.max(0, phaseWindow.rawEnd - phaseWindow.rawStart)
    const fraction = Math.max(0, Math.min(1, (simTime - phaseWindow.simStart) / simSpan))
    const rawTarget = phaseWindow.rawStart + fraction * rawSpan
    const rate = rawSpan > 0 ? Math.max(0.0625, Math.min(4, rawSpan / simSpan)) : 0.0625
    if (Math.abs(rawVideo.currentTime - rawTarget) > 0.035) rawVideo.currentTime = rawTarget
    rawVideo.playbackRate = rate
    setProgress(rawTarget / duration); setSyncRate(rate); setSyncPhase(phaseWindow.phase)
  }
  const seek = (fraction: number) => {
    const value = Math.max(0, Math.min(1, fraction)); const rawTarget = value * duration
    setProgress(value); if (videoRef.current?.duration) videoRef.current.currentTime = rawTarget
    if (phaseSync && simVideoRef.current) {
      const window = syncWindows.find((item) => rawTarget >= item.rawStart && rawTarget <= item.rawEnd)
      if (window) {
        const rawSpan = window.rawEnd - window.rawStart
        const mapped = rawSpan > 0 ? (rawTarget - window.rawStart) / rawSpan : 0
        simVideoRef.current.currentTime = window.simStart + mapped * (window.simEnd - window.simStart)
      }
    }
  }
  const togglePhaseSync = () => {
    const rawVideo = videoRef.current; const simVideo = simVideoRef.current
    if (!rawVideo || !simVideo || !syncAvailable) return
    rawVideo.pause(); simVideo.pause(); setSyncPlaying(false)
    if (phaseSync) { rawVideo.playbackRate = 1; setSyncRate(1); setSyncPhase('—'); setPhaseSync(false); return }
    setPhaseSync(true); simVideo.currentTime = syncWindows[0].simStart; rawVideo.currentTime = syncWindows[0].rawStart; setSyncPhase(syncWindows[0].phase)
  }
  const toggleSynchronizedPlayback = async () => {
    const rawVideo = videoRef.current; const simVideo = simVideoRef.current
    if (!rawVideo || !simVideo || !phaseSync) return
    if (!simVideo.paused) { simVideo.pause(); rawVideo.pause(); setSyncPlaying(false); return }
    if (simVideo.ended || simVideo.currentTime >= syncWindows.at(-1)!.simEnd) simVideo.currentTime = syncWindows[0].simStart
    synchronizeAt(simVideo.currentTime)
    try { await Promise.all([simVideo.play(), rawVideo.play()]); setSyncPlaying(true) } catch { setSyncPlaying(false) }
  }
  const selectedVideo = rawVideos.find((video) => video.name === selectedRaw)
  const startProcessing = async () => {
    setPipelineError('')
    try {
      const response = await fetch('/api/process', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ video: selectedRaw, device, config: selectedConfig }) })
      const body = await response.json()
      if (!response.ok) throw new Error(body.error ?? 'Pipeline could not be started.')
      setPipelineJob(body)
    } catch (cause) { setPipelineError(cause instanceof Error ? cause.message : String(cause)) }
  }

  return <div className="app-shell">
    <header className="topbar"><div className="brand"><div className="mark">M</div><div><strong>MIMIC</strong><span>RUN INSPECTOR</span></div></div><div className="pipeline"><span>DEMONSTRATION</span><i>→</i><span>SKILL MODEL</span><i>→</i><span>RETARGET</span><i>→</i><span>PANDA</span></div><div className="run-control"><span className="status-dot" /><label htmlFor="run-select">Artifact set</label><select id="run-select" value={runId} onChange={(e) => setRunId(e.target.value)}>{runs.map((run) => <option key={run.id} value={run.id}>{run.name} · {run.success ? 'success' : run.completed ? 'failed' : 'partial'}</option>)}</select></div></header>
    <main>
      <section className="panel process-panel"><PanelTitle eyebrow="Local pipeline" title="Process raw demonstration" end={<span className={`job-status ${pipelineJob?.status ?? 'idle'}`}><i />{pipelineJob?.status ?? 'IDLE'}</span>} /><div className="process-body"><div className="process-controls"><label><span>RAW VIDEO</span><select value={selectedRaw} onChange={(event) => setSelectedRaw(event.target.value)} disabled={pipelineJob?.status === 'running'}>{rawVideos.map((video) => <option key={video.name} value={video.name}>{video.name}{video.has_results ? ' · processed' : ' · new'}</option>)}</select></label><label><span>ROBOT CONFIG</span><select value={selectedConfig} onChange={(event) => setSelectedConfig(event.target.value)} disabled={pipelineJob?.status === 'running'}>{robotConfigs.map((config) => <option key={config.id} value={config.id}>{config.name}{config.default ? ' · default' : ''}</option>)}</select></label><label><span>INFERENCE DEVICE</span><select value={device} onChange={(event) => setDevice(event.target.value)} disabled={pipelineJob?.status === 'running'}><option value="cpu">CPU</option><option value="mps">Apple MPS</option><option value="cuda">CUDA</option></select></label><button type="button" onClick={startProcessing} disabled={!selectedRaw || !selectedConfig || pipelineJob?.status === 'running'}>{pipelineJob?.status === 'running' ? 'PROCESSING…' : selectedVideo?.has_results ? 'REPROCESS RUN' : 'PROCESS VIDEO'}<b>▶</b></button></div><div className="process-monitor"><div><strong>{pipelineJob?.stage ?? 'Ready'}</strong><span>{pipelineJob?.progress ?? 0}%</span></div><div className={`process-progress ${pipelineJob?.status ?? 'idle'}`}><i style={{ width: `${pipelineJob?.progress ?? 0}%` }} /></div><small>{pipelineError || pipelineJob?.message || 'Select a video from data/raw/.'}</small></div></div></section>
      {loading && <div className="loading"><i />INDEXING RUN ARTIFACTS…</div>}{error && <div className="error-state">{error}<small>Start with <code>npm run dev</code> from web/.</small></div>}
      {!loading && detail && <>
        <div className={`sync-strip ${phaseSync ? 'active' : ''}`}><div><span className="sync-icon">⇄</span><div><strong>Phase-aligned playback</strong><small>Simulation is the master clock; raw footage is slowed per matching phase.</small></div></div><div className="sync-readout"><span>PHASE <b>{syncPhase}</b></span><span>RAW <b>{syncRate.toFixed(2)}×</b></span><span>SIM <b>1.00×</b></span></div><button type="button" className="sync-mode" onClick={togglePhaseSync} disabled={!syncAvailable}>{phaseSync ? 'SYNC ON' : 'ENABLE SYNC'}</button><button type="button" className="sync-play" onClick={toggleSynchronizedPlayback} disabled={!phaseSync}>{syncPlaying ? 'Ⅱ PAUSE BOTH' : '▶ PLAY BOTH'}</button></div>
        <div className="overview-grid">
          <section className="panel video-panel raw-video-panel"><PanelTitle eyebrow="01 · perception input" title="Human demonstration" end={<div className="video-meta"><span>{detail.video?.frame_count ?? '—'} FRAMES</span><span>{(detail.video?.fps ?? 0).toFixed(1)} FPS</span></div>} /><div className="video-stage">
            {detail.source_video_url ? <video key={detail.source_video_url} ref={videoRef} src={detail.source_video_url} controls playsInline onTimeUpdate={(e) => { if (!phaseSync && e.currentTarget.duration) setProgress(e.currentTarget.currentTime / e.currentTarget.duration) }} onPlay={() => { if (phaseSync && simVideoRef.current?.paused) void simVideoRef.current.play() }} /> : <div className="empty-video"><span>RAW FOOTAGE NOT RETAINED</span><small>Add a basename-matched source video under data/raw/ or data/videos/.</small></div>}
            <div className={`media-badge ${detail.source_video_url ? 'source' : 'fallback'}`}><i />{detail.source_video_url ? 'RAW SOURCE FOOTAGE' : 'SOURCE ARTIFACT MISSING'}</div><div className="video-overlay"><span>RUN / {detail.name}</span><span>{time(progress * duration)}</span></div>
          </div></section>
          <section className="panel video-panel sim-video-panel"><PanelTitle eyebrow="02 · robot output" title="Robot simulation" end={<span className={`result-badge ${result?.success ? 'success' : 'failure'}`}><i />{result?.success ? 'SUCCESS' : result ? 'FAILED' : 'NO RESULT'}</span>} /><div className="video-stage">{detail.simulation_video_url ? <video key={detail.simulation_video_url} ref={simVideoRef} src={detail.simulation_video_url} controls playsInline onLoadedMetadata={(event) => setSimDuration(event.currentTarget.duration)} onTimeUpdate={(event) => { if (phaseSync) synchronizeAt(event.currentTarget.currentTime) }} onPlay={() => { if (phaseSync) { synchronizeAt(simVideoRef.current?.currentTime ?? 0); if (videoRef.current?.paused) void videoRef.current.play(); setSyncPlaying(true) } }} onPause={() => { if (phaseSync) { videoRef.current?.pause(); setSyncPlaying(false) } }} onEnded={() => { videoRef.current?.pause(); setSyncPlaying(false) }} /> : <div className="empty-video"><span>NO SIMULATION VIDEO</span><small>No `.mimic.mp4` recording exists for this run.</small></div>}<div className="media-badge source"><i />MUJOCO OUTPUT</div><div className="video-overlay"><span>SIMULATION / {detail.name}</span><span>{result?.final_position_error_m != null ? `${(result.final_position_error_m * 1000).toFixed(2)} mm error` : 'NO RESULT'}</span></div></div></section>
          <aside className="side-stack">
            <section className={`panel action-panel ${cssPhase[action?.phase ?? 'IDLE']}`}><PanelTitle eyebrow="03 · resolved state" title="Current catalog state" end={<span className="frame-id mono">FRAME {action?.frame_idx ?? '—'}</span>} /><div className="action-readout"><span className="crosshair">⌖</span><div><strong>{action?.phase ?? 'NO SIGNAL'}</strong><span>COMPOSITE SKILL LABEL</span></div></div><div className="confidence"><div><span>CLASSIFIER CONFIDENCE</span><strong>{(confidence * 100).toFixed(1)}%</strong></div><div className="meter"><i style={{ width: `${confidence * 100}%` }} /></div></div><div className="score-list">{phases.map((phase) => <div key={phase}><span>{phase}</span><i><b style={{ width: `${(scores[phase] ?? 0) * 100}%` }} /></i><em>{((scores[phase] ?? 0) * 100).toFixed(0)}</em></div>)}</div></section>
          </aside>
        </div>
        <Timeline segments={segments} phases={phases} duration={duration} progress={progress} onSeek={seek} />
        <div className="bottom-grid">
          <section className="panel trajectory-panel"><PanelTitle eyebrow="04 · coordinate retargeting" title="Object trajectory · world XY" end={<span className="coordinate mono">METERS / PANDA BASE</span>} /><Trajectory detail={detail} /></section>
          <section className="panel result-panel"><PanelTitle eyebrow="05 · MuJoCo verification" title="Robot execution" end={<span className={`result-badge ${result?.success ? 'success' : 'failure'}`}><i />{result?.success ? 'SUCCESS' : result ? 'FAILED' : 'NO RESULT'}</span>} /><div className="result-body"><div className="metrics"><div><span>GRASP</span><strong className={result?.grasp_occurred ? 'ok' : ''}>{result?.grasp_occurred ? 'YES' : 'NO'}</strong></div><div><span>TRANSPORT</span><strong className={result?.transported ? 'ok' : ''}>{result?.transported ? 'YES' : 'NO'}</strong></div><div><span>RELEASE</span><strong className={result?.released ? 'ok' : ''}>{result?.released ? 'YES' : 'NO'}</strong></div><div className="error-metric"><span>FINAL POSITION ERROR</span><strong>{result?.final_position_error_m != null ? `${(result.final_position_error_m * 1000).toFixed(2)} mm` : '—'}</strong></div></div><div className="event-log"><div className="event-log-head"><span>EXECUTION TRACE</span><span>{detail.execution.transitions.length} TRANSITIONS</span></div>{detail.execution.transitions.filter((item, index, all) => index === 0 || all[index - 1].skill !== item.skill).slice(-5).map((item) => <div key={`${item.step}-${item.skill}`}><time>{item.timestamp_s.toFixed(2)}s</time><i className={cssPhase[item.phase] ?? 'idle'} /><span>{item.skill.replaceAll('_', ' ')}</span><em>STEP {item.step}</em></div>)}</div>{result?.failure && <div className="failure-note">{result.failure}</div>}</div></section>
        </div>
      </>}
    </main>
    <footer><span>MIMIC / VISUALIZATION LAYER</span><span>READ-ONLY · RESULTS/</span><span>{detail?.catalog?.fingerprint ? `CATALOG ${detail.catalog.fingerprint.slice(0, 8)}` : 'NO CATALOG'}</span></footer>
  </div>
}
