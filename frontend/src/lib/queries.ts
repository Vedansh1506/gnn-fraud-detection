/**
 * Server state, via TanStack Query.
 *
 * One place decides how often each thing is refetched, because polling cadence
 * is a product decision rather than a per-component detail: the queue is a live
 * worklist and should move on its own; model metrics are artifacts on disk and
 * effectively static within a session.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseQueryOptions,
} from '@tanstack/react-query'
import { api, type AnalystDecision, type FlagsResponse } from './api'
import { useAuth, useToken } from './auth'

export const queryKeys = {
  health: ['health'] as const,
  flags: (status: string, minScore: number) => ['flags', status, minScore] as const,
  graph: (accountKey: string, hops: number) => ['graph', accountKey, hops] as const,
  models: ['models'] as const,
}

/** Polled: it drives the degraded-mode banner, which must not go stale. */
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => api.health(),
    refetchInterval: 20_000,
    retry: 1,
  })
}

export function useFlags(
  status: string,
  minScore: number,
  options?: Partial<UseQueryOptions<FlagsResponse>>,
) {
  const token = useToken()
  const { handleUnauthorized } = useAuth()

  return useQuery<FlagsResponse>({
    queryKey: queryKeys.flags(status, minScore),
    queryFn: async () => {
      try {
        return await api.flags(token, { status, minScore, limit: 200 })
      } catch (error) {
        handleUnauthorized(error)
        throw error
      }
    },
    enabled: Boolean(token),
    // The stream is live, so the queue should fill in without a manual refresh.
    refetchInterval: 15_000,
    ...options,
  })
}

/**
 * Neo4j is the only dependency the dashboard needs that scoring does not, so
 * this query is allowed to fail on its own: the detail screen renders the SHAP
 * explanation regardless and shows "graph view unavailable" in the panel (§7).
 */
export function useGraph(accountKey: string | null, hops = 2) {
  const token = useToken()
  return useQuery({
    queryKey: queryKeys.graph(accountKey ?? '', hops),
    queryFn: () => api.graph(token, accountKey as string, hops),
    enabled: Boolean(token && accountKey),
    retry: 0, // A down graph store won't come back within a retry window.
    staleTime: 5 * 60_000,
  })
}

export function useModels() {
  const token = useToken()
  const { handleUnauthorized } = useAuth()

  return useQuery({
    queryKey: queryKeys.models,
    queryFn: async () => {
      try {
        return await api.models(token)
      } catch (error) {
        handleUnauthorized(error)
        throw error
      }
    },
    enabled: Boolean(token),
    staleTime: 60_000,
  })
}

export function useSubmitFeedback() {
  const token = useToken()
  const client = useQueryClient()

  return useMutation({
    mutationFn: ({ txId, decision }: { txId: string; decision: AnalystDecision }) =>
      api.feedback(token, txId, decision),
    onSuccess: () => {
      // A decided flag leaves the open queue and changes the summary counts,
      // so both have to be refetched - not just the list.
      client.invalidateQueries({ queryKey: ['flags'] })
      client.invalidateQueries({ queryKey: queryKeys.models })
    },
  })
}

export function useTriggerRetrain() {
  const token = useToken()
  const client = useQueryClient()

  return useMutation({
    mutationFn: () => api.retrain(token),
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.models }),
  })
}
