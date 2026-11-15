// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The entry: a quality-ordered free-model router. It reads the brief (provider directory, quality
// table, workload manifest), probes each provider's catalog route, builds one catalog, registers
// it with the env, and serves the client's requests with the best free model that can serve each
// one, failing over across providers and wires when a provider refuses.
//
//   - quality: every free candidate that meets a request's requirements is ranked by the quality
//     table's rating_milli (ties: the simplest upstream wire, then provider and model id), and the
//     request is served by the best candidate that answers.
//   - failover: a 402, 429 or 5xx (or an unreachable provider) moves to the next candidate; a 402
//     marks that provider spent for the rest of the trial. A 2xx answer is always used, since a
//     second successful call would break the trial's one-provider rule.
//   - wires: all six client/provider combinations are served — OpenAI and Anthropic clients
//     against OpenAI, Anthropic and Gemini providers, in JSON and in server-sent events — with the
//     request's system prompt, tools, images, stop sequences, max_tokens and JSON mode carried
//     through untouched, and the provider's text, usage and stop reason mapped back into the
//     client's wire. A JSON-mode request never routes to an anthropic-wire provider, which cannot
//     carry response_format; streaming OpenAI requests set stream_options.include_usage so the
//     usage chunk arrives.
//   - catalog: the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row.
//
// Everything is a pure function of the brief, the requests and the providers' answers: no clock
// and no unseeded randomness, so a replay leaves the same state export.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   providers.ts       the brief's types and parser, the auth headers, the catalog probe.
//   wire.ts            Anthropic <-> Chat Completions translation, Gemini translation, SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
  type Wire,
} from './providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  normalizeAnthropic,
  normalizeOpenAI,
  normalizedToGemini,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  type ServedModel,
} from './wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/** Providers whose budget is certainly spent: a 402 names the one provider that signals it that way. */
const deadProviders = new Set<string>();

/**
 * Every free model that can serve one request, best first: the quality table's rating_milli, then the
 * simplest upstream wire on a tie (fewer translations between the client and the provider), then
 * provider and model id so the order is fixed. A JSON-mode request never routes to an anthropic-wire
 * provider: that wire cannot carry response_format, so the provider would echo json_mode false.
 */
function candidatesFor(requires: Requirement): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      if (requires.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const wireRank = (wire: Wire): number => (wire === 'openai' ? 0 : wire === 'anthropic' ? 1 : 2);
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      wireRank(a.probe.provider.wire) - wireRank(b.probe.provider.wire) ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the manifest is the declared source of each request's requirements; this fallback is a
  // guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** How long one provider call may take before the next candidate is tried, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 15000;

/** The upstream request for one candidate: its URL, headers and JSON body. */
function upstreamFor(route: Route, clientApi: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; payload: Json } {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  const wire = route.probe.provider.wire;
  if (wire === 'openai') {
    const payload = clientApi === 'openai' ? { ...body, model: route.model.id } : anthropicToOpenAI(body, route.model.id);
    if (stream && !isRecord(payload.stream_options)) payload.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, payload };
  }
  if (wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const payload = clientApi === 'openai' ? openAIToAnthropicRequest(body, route.model.id) : { ...body, model: route.model.id };
    return { url: `${base}/v1/messages`, headers, payload };
  }
  const normalized = clientApi === 'openai' ? normalizeOpenAI(body) : normalizeAnthropic(body);
  const path = stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(route.model.id)}${path}`, headers, payload: normalizedToGemini(normalized) };
}

/** POST one request to one candidate provider. */
async function callProvider(route: Route, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const { url, headers, payload } = upstreamFor(route, clientApi, body);
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(payload), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Answer the client from one provider's 2xx answer, translated into the client's wire. */
async function sendUpstream(clientApi: 'openai' | 'anthropic', route: Route, body: Json, upstream: Response, res: ServerResponse): Promise<void> {
  const wire = route.probe.provider.wire;
  const model = route.model.id;
  const stream = body.stream === true;
  const sseHeaders = { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' };
  if (!stream) {
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer)) return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'the provider answered a body that is not one JSON object');
    if (clientApi === 'openai') {
      if (wire === 'openai') {
        const text = JSON.stringify(answer);
        res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
        res.end(text);
      } else if (wire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(answer, model));
      else sendJson(res, 200, geminiToOpenAI(answer, model));
      return;
    }
    if (wire === 'anthropic') {
      const text = JSON.stringify(answer);
      res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
      res.end(text);
    } else if (wire === 'openai') {
      const requested = typeof body.model === 'string' && body.model !== '' ? body.model : model;
      sendJson(res, 200, openAIToAnthropic(answer, requested));
    } else sendJson(res, 200, geminiToAnthropic(answer, model));
    return;
  }
  if (upstream.body === null) return sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_error', 'the provider answered no stream');
  res.writeHead(200, sseHeaders);
  if (clientApi === 'openai' && wire === 'openai') {
    await pipeStream(upstream.body, res);
    return;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    await pipeStream(upstream.body, res);
    return;
  }
  const events =
    clientApi === 'openai'
      ? wire === 'anthropic'
        ? openAIEventsFromAnthropic(upstream.body, model)
        : openAIEventsFromGemini(upstream.body, model)
      : wire === 'openai'
        ? anthropicEventsFromOpenAI(upstream.body, typeof body.model === 'string' && body.model !== '' ? body.model : model)
        : anthropicEventsFromGemini(upstream.body, model);
  for await (const event of events) res.write(event);
  res.end();
}

/**
 * Serve one client request: try the candidates best-first and answer from the first provider that
 * answers 2xx. A refusal (402, 429, 5xx) or an unreachable provider costs one call and moves to the
 * next candidate; a 402 marks that provider spent for the rest of the trial, since only a spent
 * budget is signalled that way. A 2xx answer is always used: a second successful call would break
 * the trial's one-provider rule.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 0;
  for (const route of candidates) {
    if (deadProviders.has(route.probe.provider.id)) continue;
    let upstream: Response;
    try {
      upstream = await callProvider(route, clientApi, body);
    } catch (err) {
      log(`provider ${route.probe.provider.id}/${route.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      lastStatus = upstream.status;
      if (upstream.status === 402) deadProviders.add(route.probe.provider.id);
      log(`provider ${route.probe.provider.id}/${route.model.id} refused HTTP ${upstream.status}`);
      await upstream.text().catch(() => undefined);
      continue;
    }
    await sendUpstream(clientApi, route, body, upstream, res);
    return;
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', `every free model refused this request (last HTTP ${lastStatus})`);
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that serves it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
