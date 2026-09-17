/**
 * Typed client for the FastAPI scoring service.
 *
 * The types here mirror `src/api/schemas.py` by hand. That duplication is
 * deliberate for an MVP - generating them from the OpenAPI schema is the right
 * answer once the contract moves often, but a generator is another build step
 * to keep alive. The contract tests on the Python side are what stop these
 * drifting silently; if a field here is wrong, the UI shows `undefined` rather
 * than lying, which is the failure mode we want.
 */

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export interface ExplanationFactor {
  feature: string
  contribution: number
  plain: string
}

export type AnalystDecision = 'confirmed_fraud' | 'false_positive'

export interface FlaggedTransaction {
  tx_id: string
  account_key: string
  score: number
  is_flagged: boolean
  model_version: string
  embedding_version: string
  scored_at: string
  explanation: ExplanationFactor[]
  receiver_account_key: string | null
  amount_paid: number | null
  payment_currency: string | null
  payment_format: string | null
  tx_timestamp: string | null
  analyst_decision: AnalystDecision | null
  decided_at: string | null
  decided_by: string | null
}

export interface QueueSummary {
  open_flags: number
  confirmed_today: number
  dismissed_today: number
  model_version: string
}

export interface FlagsResponse {
  flags: FlaggedTransaction[]
  count: number
  summary: QueueSummary
}

export interface GraphEdge {
  source: string
  target: string
  tx_id: string
  amount_paid: number
  is_laundering: boolean
}

export interface GraphResponse {
  account_key: string
  hops: number
  nodes: string[]
  edges: GraphEdge[]
  truncated: boolean
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unhealthy'
  model_version: string
  embedding_version: string
  components: {
    model_loaded: boolean
    database_reachable: boolean
    embeddings_loaded: boolean
  }
}

export interface ModelVersion {
  version: string
  auprc: number
  roc_auc: number
  test_rows: number
  test_positives: number
  best_f1_threshold: number
  best_f1_precision: number
  best_f1_recall: number
  best_f1: number
  is_serving: boolean
  embedding_version: string | null
  dropped_features: string[]
}

export interface RetrainRun {
  job_id: string
  status: string
  requested_by: string
  requested_at: string
  feedback_rows_pending: number
  resulting_model_version: string | null
}

export interface ModelsResponse {
  serving_model_version: string
  serving_embedding_version: string
  flag_threshold: number
  versions: ModelVersion[]
  baseline_auprc: number | null
  gnn_auprc: number | null
  auprc_lift_pct: number | null
  override_rate: number | null
  reviewed_count: number
  drift: {
    state: 'ok' | 'warning' | 'alert' | 'not_instrumented'
    message: string
    checked_at: string | null
  }
  recent_retrains: RetrainRun[]
}

export interface LoginResponse {
  access_token: string
  token_type: string
  role: string
  expires_in_minutes: number
}

export interface RetrainResponse {
  job_id: string
  status: string
  instructions: string
  requested_at: string
  feedback_rows_pending: number
}

/**
 * An API failure the UI can act on.
 *
 * `kind` exists so screens can tell apart the three cases that need genuinely
 * different treatment: the session expired (send them to login), the service is
 * unreachable (show the plain "can't reach the scoring service" message), or
 * one dependency is down (degrade that panel only).
 */
export class ApiError extends Error {
  readonly status: number
  readonly kind: 'unauthorized' | 'forbidden' | 'unavailable' | 'network' | 'other'

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.kind =
      status === 401
        ? 'unauthorized'
        : status === 403
          ? 'forbidden'
          : status === 503
            ? 'unavailable'
            : status === 0
              ? 'network'
              : 'other'
  }
}

/** Plain, actionable message for any failure - never a stack trace (§7). */
export function messageFor(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.kind) {
      case 'network':
        return "Can't reach the scoring service. Check it's running on " + API_BASE_URL + '.'
      case 'unauthorized':
        return 'Your session has expired. Sign in again to continue.'
      case 'forbidden':
        return 'Your account does not have permission to do that.'
      default:
        return error.message
    }
  }
  return 'Something went wrong. Check the service logs for detail.'
}

async function request<T>(
  path: string,
  { token, method = 'GET', body }: { token?: string | null; method?: string; body?: unknown } = {},
): Promise<T> {
  const headers: Record<string, string> = {}
  if (token) headers.Authorization = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  } catch {
    // fetch only rejects for network-level failures, which is exactly the
    // "API is down" case - status 0 marks it as distinct from an HTTP error.
    throw new ApiError('Network request failed', 0)
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`
    try {
      const payload = await response.json()
      if (typeof payload?.detail === 'string') detail = payload.detail
    } catch {
      // A non-JSON error body is not worth failing over; the status carries it.
    }
    throw new ApiError(detail, response.status)
  }

  return (await response.json()) as T
}

export const api = {
  login: (username: string, password: string) =>
    request<LoginResponse>('/auth/login', { method: 'POST', body: { username, password } }),

  health: () => request<HealthResponse>('/health'),

  flags: (token: string, params: { status?: string; limit?: number; minScore?: number } = {}) => {
    const query = new URLSearchParams()
    if (params.status) query.set('status', params.status)
    if (params.limit) query.set('limit', String(params.limit))
    if (params.minScore) query.set('min_score', String(params.minScore))
    return request<FlagsResponse>(`/flags?${query.toString()}`, { token })
  },

  graph: (token: string, accountKey: string, hops = 2) =>
    request<GraphResponse>(`/graph/${encodeURIComponent(accountKey)}?hops=${hops}`, { token }),

  feedback: (token: string, txId: string, decision: AnalystDecision) =>
    request<{ status: string; tx_id: string }>('/feedback', {
      token,
      method: 'POST',
      body: { tx_id: txId, analyst_decision: decision },
    }),

  models: (token: string) => request<ModelsResponse>('/models', { token }),

  retrain: (token: string) => request<RetrainResponse>('/retrain', { token, method: 'POST' }),
}
