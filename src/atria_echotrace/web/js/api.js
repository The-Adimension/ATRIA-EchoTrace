/**
 * Typed-ish API client for the ATRIA EchoTrace backend.
 *
 * Every backend error arrives as an RFC 7807-style problem document
 * ({type, title, status, detail}); `ApiError` preserves both the human-facing
 * detail and the status so callers can distinguish "model not loaded" (409) from
 * "AI tier absent" (503) without string matching.
 *
 * This client intentionally mirrors the *complete* HTTP surface, including five helpers
 * the SPA does not call: `datasetReport` (covered by `atria doctor`), `frameMetrics`
 * (the UI uses the study-level `metrics`), and `evaluationRuns` / `startEvaluation` /
 * `evaluationRun` — evaluation is a long batch job that belongs to a CLI-shaped stage,
 * see the Stages panel. They are a maintained client for scripting, not dead code.
 */

export class ApiError extends Error {
  constructor(message, status, title) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.title = title || 'Error';
  }
}

async function request(path, { method = 'GET', body, signal, formData } = {}) {
  const init = { method, signal, headers: {} };

  if (formData) {
    init.body = formData; // Browser sets the multipart boundary itself.
  } else if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(path, init);
  } catch (cause) {
    if (cause.name === 'AbortError') throw cause;
    throw new ApiError(
      `Cannot reach the ATRIA server at ${location.origin}. Is it still running?`,
      0,
      'Network error'
    );
  }

  if (response.status === 204) return null;

  const isJson = (response.headers.get('content-type') || '').includes('json');
  const payload = isJson ? await response.json().catch(() => null) : null;

  if (!response.ok) {
    const detail =
      (payload && (payload.detail || payload.title)) ||
      `${response.status} ${response.statusText}`;
    throw new ApiError(detail, response.status, payload && payload.title);
  }
  return payload;
}

export const api = {
  // --- meta ---
  capabilities: () => request('/api/meta/capabilities'),
  disclaimers: () => request('/api/meta/disclaimers'),
  datasetReport: () => request('/api/meta/dataset-report'),
  // The lifecycle-stage registry, resolved server-side. Stage titles, descriptions and
  // commands live only in api/stages.py — never duplicated here or in app.js.
  stages: () => request('/api/meta/stages'),

  // --- dataset ---
  cases: ({ source, view } = {}) => {
    const params = new URLSearchParams();
    if (source) params.set('source', source);
    if (view) params.set('view', view);
    const query = params.toString();
    return request(`/api/dataset/cases${query ? `?${query}` : ''}`);
  },
  caseDetail: (caseKey) => request(`/api/dataset/cases/${encodeURIComponent(caseKey)}`),
  uploadFrame: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request('/api/dataset/uploads', { method: 'POST', formData });
  },

  // --- model lifecycle ---
  modelStatus: () => request('/api/model/status'),
  loadModel: (adapter) => request('/api/model/load', { method: 'POST', body: { adapter } }),
  unloadModel: () => request('/api/model/unload', { method: 'POST' }),

  // --- inference ---
  predict: (payload, signal) =>
    request('/api/inference/predict', { method: 'POST', body: payload, signal }),

  // --- clinical metrics ---
  metrics: (payload) => request('/api/clinical/metrics', { method: 'POST', body: payload }),
  frameMetrics: (payload) =>
    request('/api/clinical/frame-metrics', { method: 'POST', body: payload }),

  // --- revisions ---
  createRevision: (payload) => request('/api/revisions', { method: 'POST', body: payload }),
  revisions: () => request('/api/revisions'),
  revision: (revisionId) => request(`/api/revisions/${encodeURIComponent(revisionId)}`),

  // --- evaluation ---
  evaluationRuns: () => request('/api/evaluation/runs'),
  startEvaluation: (payload) =>
    request('/api/evaluation/runs', { method: 'POST', body: payload }),
  evaluationRun: (runId) => request(`/api/evaluation/runs/${encodeURIComponent(runId)}`),
};
