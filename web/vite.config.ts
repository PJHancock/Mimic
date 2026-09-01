import react from '@vitejs/plugin-react'
import { createReadStream, existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { extname, join, relative, resolve, sep } from 'node:path'
import { spawn } from 'node:child_process'
import { defineConfig, type Plugin } from 'vite'

const projectRoot = resolve(import.meta.dirname, '..')
const resultsRoot = join(projectRoot, 'results')
type Json = Record<string, any>

type PipelineJob = {
  id: string | null
  status: 'idle' | 'running' | 'succeeded' | 'failed'
  progress: number
  stage: string
  message: string
  video: string | null
  run_id: string | null
  logs: string[]
  exit_code: number | null
  started_at: string | null
  finished_at: string | null
}

let pipelineJob: PipelineJob = {
  id: null, status: 'idle', progress: 0, stage: 'Ready', message: 'Select a raw video to process.',
  video: null, run_id: null, logs: [], exit_code: null, started_at: null, finished_at: null,
}
let artifactTimer: ReturnType<typeof setInterval> | null = null

function readJson(path?: string): Json | null {
  return path && existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : null
}

function findFiles(root: string, depth = 3): string[] {
  if (!existsSync(root) || depth < 0) return []
  return readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const path = join(root, entry.name)
    return entry.isDirectory() ? findFiles(path, depth - 1) : [path]
  })
}

function artifactUrl(path?: string): string | null {
  if (!path) return null
  const encoded = relative(projectRoot, path).split(sep).map(encodeURIComponent).join('/')
  const version = Math.trunc(statSync(path).mtimeMs)
  return `/artifacts/${encoded}?v=${version}`
}

function executionSummary(path?: string) {
  if (!path || !existsSync(path)) return { result: null, transitions: [], samples: [] }
  const events = readFileSync(path, 'utf8').split('\n').filter(Boolean).flatMap((line) => {
    try { return [JSON.parse(line)] } catch { return [] }
  })
  const allSamples = events.filter((event) => event.event === 'sample')
  const stride = Math.max(1, Math.ceil(allSamples.length / 220))
  return {
    result: events.findLast((event) => event.event === 'result') ?? null,
    transitions: events.filter((event) => event.event === 'transition'),
    samples: allSamples.filter((_: Json, index: number) => index % stride === 0 || index === allSamples.length - 1).map((sample: Json) => ({
      timestamp_s: sample.state?.timestamp_s,
      phase: sample.phase,
      skill: sample.skill,
      object_position: sample.state?.object_position,
      tool_position: sample.state?.tool_pose?.position,
      position_error_m: sample.ik?.position_error_m,
    })),
  }
}

function sourceVideoFor(stem: string) {
  return [join(projectRoot, 'data', 'raw'), join(projectRoot, 'data', 'videos')]
    .flatMap((root) => findFiles(root, 4))
    .find((path) => ['.mp4', '.mov', '.m4v', '.webm'].includes(extname(path).toLowerCase())
      && path.split(sep).at(-1)!.replace(/\.[^.]+$/, '') === stem)
}

function rawVideoRecords() {
  const root = join(projectRoot, 'data', 'raw')
  return findFiles(root, 1)
    .filter((path) => ['.mp4', '.mov', '.m4v', '.webm'].includes(extname(path).toLowerCase()))
    .map((path) => {
      const name = path.split(sep).at(-1)!
      const stem = name.replace(/\.[^.]+$/, '')
      const resultDir = join(resultsRoot, stem)
      return {
        name,
        stem,
        size_bytes: statSync(path).size,
        modified_ms: statSync(path).mtimeMs,
        has_results: existsSync(join(resultDir, `${stem}_task_input.json`)),
      }
    })
    .sort((a, b) => b.modified_ms - a.modified_ms)
}

function applyPipelineMilestone(line: string) {
  const milestones: [RegExp, number, string][] = [
    [/Processing .* for robot/, 4, 'Preparing pipeline'],
    [/1\. Extracting object position tracks/, 10, 'Tracking object'],
    [/Extracted \d+ position samples/, 22, 'Object tracking complete'],
    [/2\. Extracting frame features/, 28, 'Extracting visual features'],
    [/Extracted \d+ embeddings/, 46, 'Visual features complete'],
    [/3\. Predicting action sequences/, 52, 'Classifying skill states'],
    [/Predicted complete scores/, 66, 'Classification complete'],
    [/4\. Building consolidated robot task input/, 70, 'Building task input'],
    [/5\. Saving results/, 74, 'Writing inference artifacts'],
    [/Wrote episode .* path points/, 84, 'World waypoints ready'],
    [/Wrote simulation video to/, 96, 'Simulation recording complete'],
  ]
  for (const [pattern, progress, stage] of milestones) {
    if (pattern.test(line) && progress >= pipelineJob.progress) {
      pipelineJob.progress = progress
      pipelineJob.stage = stage
      pipelineJob.message = line.replace(/^\s*[✓⚠✗]?\s*/, '')
    }
  }
}

function appendPipelineOutput(chunk: Buffer) {
  for (const line of chunk.toString().split(/\r?\n/).map((value) => value.trim()).filter(Boolean)) {
    pipelineJob.logs = [...pipelineJob.logs.slice(-79), line]
    applyPipelineMilestone(line)
  }
}

function watchPipelineArtifacts(stem: string) {
  if (artifactTimer) clearInterval(artifactTimer)
  const dir = join(resultsRoot, stem)
  const startedMs = pipelineJob.started_at ? Date.parse(pipelineJob.started_at) : Date.now()
  const isFresh = (path: string) => existsSync(path) && statSync(path).mtimeMs >= startedMs - 1_000
  artifactTimer = setInterval(() => {
    if (pipelineJob.status !== 'running') return
    const task = join(dir, `${stem}_task_input.json`)
    const scores = join(dir, `${stem}_scores.json`)
    const waypoints = join(dir, `${stem}_world_waypoints.json`)
    const execution = join(dir, `${stem}_execution.jsonl`)
    if (isFresh(task) && isFresh(scores) && pipelineJob.progress < 76) {
      pipelineJob.progress = 76; pipelineJob.stage = 'Inference artifacts ready'; pipelineJob.message = 'Scores and resolved task input written.'
    }
    if (isFresh(waypoints) && pipelineJob.progress < 84) {
      pipelineJob.progress = 84; pipelineJob.stage = 'World waypoints ready'; pipelineJob.message = 'Retargeted Panda waypoints written.'
    }
    if (isFresh(execution) && statSync(execution).size > 0 && pipelineJob.progress < 90) {
      pipelineJob.progress = 90; pipelineJob.stage = 'Running robot simulation'; pipelineJob.message = 'MuJoCo execution is in progress.'
    }
  }, 500)
}

function robotConfigRecords() {
  const root = join(projectRoot, 'configs', 'robots', 'panda')
  return findFiles(root, 1)
    .filter((path) => ['.yaml', '.yml'].includes(extname(path).toLowerCase()))
    .flatMap((path) => {
      const text = readFileSync(path, 'utf8')
      if (!/(?:^|\n)robot_execution:/.test(text) || /(?:^|\n)\s*model_path:\s*null\b/.test(text)) return []
      const id = path.split(sep).at(-1)!.replace(/\.ya?ml$/i, '')
      return [{ id, name: id, default: id === 'slow' }]
    })
    .sort((a, b) => Number(b.default) - Number(a.default) || a.name.localeCompare(b.name))
}

function startPipeline(videoName: string, device: string, configId: string) {
  if (pipelineJob.status === 'running') throw new Error('A pipeline job is already running.')
  const record = rawVideoRecords().find((video) => video.name === videoName)
  if (!record) throw new Error('Selected video is not available under data/raw/.')
  if (!['cpu', 'mps', 'cuda'].includes(device)) throw new Error('Unsupported inference device.')
  const config = robotConfigRecords().find((item) => item.id === configId)
  if (!config) throw new Error('Selected robot config is not available.')
  const videoPath = join(projectRoot, 'data', 'raw', record.name)
  pipelineJob = {
    id: `${Date.now()}`,
    status: 'running',
    progress: 2,
    stage: 'Starting',
    message: `Launching Mimic for ${record.name}`,
    video: record.name,
    run_id: record.stem,
    logs: [],
    exit_code: null,
    started_at: new Date().toISOString(),
    finished_at: null,
  }
  const child = spawn('uv', ['run', '--group', 'robot', 'mimic', '--video', videoPath, '--robot', 'panda', '--device', device, '--config', config.id], {
    cwd: projectRoot,
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  child.stdout.on('data', appendPipelineOutput)
  child.stderr.on('data', appendPipelineOutput)
  child.on('error', (error) => {
    pipelineJob.status = 'failed'; pipelineJob.stage = 'Launch failed'; pipelineJob.message = error.message
    pipelineJob.logs = [...pipelineJob.logs, error.message]; pipelineJob.finished_at = new Date().toISOString()
    if (artifactTimer) clearInterval(artifactTimer)
  })
  child.on('close', (code) => {
    pipelineJob.exit_code = code
    pipelineJob.finished_at = new Date().toISOString()
    pipelineJob.status = code === 0 ? 'succeeded' : 'failed'
    pipelineJob.progress = code === 0 ? 100 : pipelineJob.progress
    pipelineJob.stage = code === 0 ? 'Pipeline complete' : 'Pipeline failed'
    pipelineJob.message = code === 0 ? 'All inference and simulation artifacts are ready.' : pipelineJob.logs.at(-1) ?? `Process exited with code ${code}`
    if (artifactTimer) clearInterval(artifactTimer)
  })
  watchPipelineArtifacts(record.stem)
  return pipelineJob
}

async function readRequestJson(req: IncomingMessage): Promise<Json> {
  const chunks: Buffer[] = []
  for await (const chunk of req) {
    chunks.push(Buffer.from(chunk))
    if (chunks.reduce((total, value) => total + value.length, 0) > 65_536) throw new Error('Request body is too large.')
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')
}

function runRecords() {
  if (!existsSync(resultsRoot)) return []
  return readdirSync(resultsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name !== 'archive')
    .flatMap((entry) => {
      const dir = join(resultsRoot, entry.name)
      const files = readdirSync(dir).map((name) => join(dir, name)).filter((path) => statSync(path).isFile())
      const pick = (suffix: string) => files.find((path) => path.endsWith(suffix))
      const taskPath = pick('_task_input.json'); const scoresPath = pick('_scores.json')
      if (!taskPath && !scoresPath) return []
      const stem = (taskPath ?? scoresPath)!.split(sep).at(-1)!.replace(/_(task_input|scores)\.json$/, '')
      const executionPath = pick('_execution.jsonl')
      const simulationVideo = files.find((path) => path.endsWith(`${stem}.mimic.mp4`))
        ?? files.find((path) => path.includes('.mimic.') && path.endsWith('.mp4'))
      const { result } = executionSummary(executionPath)
      return [{
        id: entry.name,
        name: stem,
        completed: Boolean(result),
        success: result?.success ?? null,
        duration_s: readJson(taskPath)?.video?.duration_s ?? null,
        modified_ms: Math.max(...files.map((path) => statSync(path).mtimeMs)),
        artifacts: files.length,
        source_video_available: Boolean(sourceVideoFor(stem)),
        simulation_video_available: Boolean(simulationVideo),
      }]
    })
    .sort((a, b) => Number(b.success) - Number(a.success) || b.modified_ms - a.modified_ms)
}

function runDetail(id: string) {
  if (!/^[\w.-]+$/.test(id)) return null
  const dir = join(resultsRoot, id)
  if (!existsSync(dir)) return null
  const files = readdirSync(dir).map((name) => join(dir, name)).filter((path) => statSync(path).isFile())
  const pick = (suffix: string) => files.find((path) => path.endsWith(suffix))
  const taskPath = pick('_task_input.json'); const scoresPath = pick('_scores.json')
  const task = readJson(taskPath); const scores = readJson(scoresPath)
  const stem = (taskPath ?? scoresPath)?.split(sep).at(-1)?.replace(/_(task_input|scores)\.json$/, '') ?? id
  const sourceVideo = sourceVideoFor(stem)
  const simulationVideo = files.find((path) => path.endsWith(`${stem}.mimic.mp4`))
    ?? files.find((path) => path.includes('.mimic.') && path.endsWith('.mp4'))
  return {
    id,
    name: stem,
    video: task?.video ?? null,
    catalog: task?.catalog ?? scores?.catalog ?? null,
    resolved_actions: task?.resolved_actions ?? [],
    score_frames: scores?.frames ?? [],
    waypoints: readJson(pick('_world_waypoints.json')),
    execution: executionSummary(pick('_execution.jsonl')),
    source_video_url: artifactUrl(sourceVideo),
    simulation_video_url: artifactUrl(simulationVideo),
    playback_video_url: artifactUrl(sourceVideo ?? simulationVideo),
    playback_kind: sourceVideo ? 'source' : simulationVideo ? 'simulation_fallback' : 'none',
    artifact_names: files.map((path) => path.split(sep).at(-1)),
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json; charset=utf-8')
  res.end(JSON.stringify(body))
}

function serveArtifact(req: IncomingMessage, res: ServerResponse, relativePath: string) {
  const decoded = relativePath.split('/').map(decodeURIComponent).join(sep)
  const path = resolve(projectRoot, decoded)
  if (!path.startsWith(`${projectRoot}${sep}`) || !existsSync(path) || !statSync(path).isFile()) {
    return sendJson(res, 404, { error: 'Artifact not found' })
  }
  const size = statSync(path).size
  const range = req.headers.range
  const types: Record<string, string> = { '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm', '.json': 'application/json', '.jsonl': 'application/x-ndjson' }
  res.setHeader('Content-Type', types[extname(path).toLowerCase()] ?? 'application/octet-stream')
  res.setHeader('Accept-Ranges', 'bytes')
  if (['.mp4', '.mov', '.m4v', '.webm'].includes(extname(path).toLowerCase())) {
    res.setHeader('Cache-Control', 'private, max-age=0, must-revalidate')
  }
  if (!range) {
    res.setHeader('Content-Length', String(size))
    return createReadStream(path).pipe(res)
  }
  const [startText, endText] = range.replace('bytes=', '').split('-')
  const start = Number(startText); const end = endText ? Number(endText) : size - 1
  if (!Number.isFinite(start) || start < 0 || end >= size || start > end) { res.statusCode = 416; return res.end() }
  res.statusCode = 206
  res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`)
  res.setHeader('Content-Length', String(end - start + 1))
  return createReadStream(path, { start, end }).pipe(res)
}

function artifactsPlugin(): Plugin {
  return { name: 'mimic-results-adapter', configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const url = new URL(req.url ?? '/', 'http://localhost')
      if (url.pathname === '/api/raw-videos' && req.method === 'GET') return sendJson(res, 200, rawVideoRecords())
      if (url.pathname === '/api/robot-configs' && req.method === 'GET') return sendJson(res, 200, robotConfigRecords())
      if (url.pathname === '/api/process' && req.method === 'GET') return sendJson(res, 200, pipelineJob)
      if (url.pathname === '/api/process' && req.method === 'POST') {
        void readRequestJson(req).then((body) => {
          try {
            sendJson(res, 202, startPipeline(String(body.video ?? ''), String(body.device ?? 'cpu'), String(body.config ?? '')))
          } catch (error) {
            sendJson(res, 409, { error: error instanceof Error ? error.message : String(error) })
          }
        }).catch((error) => sendJson(res, 400, { error: error instanceof Error ? error.message : String(error) }))
        return
      }
      if (url.pathname === '/api/runs') return sendJson(res, 200, runRecords())
      if (url.pathname.startsWith('/api/runs/')) {
        const detail = runDetail(decodeURIComponent(url.pathname.slice('/api/runs/'.length)))
        return sendJson(res, detail ? 200 : 404, detail ?? { error: 'Run not found' })
      }
      if (url.pathname.startsWith('/artifacts/')) return serveArtifact(req, res, url.pathname.slice('/artifacts/'.length))
      next()
    })
  } }
}

export default defineConfig({
  plugins: [react(), artifactsPlugin()],
  server: { port: 4173, strictPort: false },
  preview: { port: 4173, strictPort: false },
})
