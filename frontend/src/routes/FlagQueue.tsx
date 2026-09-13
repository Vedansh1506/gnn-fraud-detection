/**
 * Flag Queue - the analyst's home screen (Design Doc §4.2).
 *
 * Top to bottom: header with a live count, the degraded-mode banner slot, the
 * summary strip, then the queue itself. Opening a flag slides the detail panel
 * in over the right-hand side rather than navigating away, so the analyst keeps
 * their place in the list - the queue is a worklist, and losing your position
 * in it after every decision is the main way these tools become tedious.
 */

import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import type { FlaggedTransaction } from '../lib/api'
import { messageFor } from '../lib/api'
import { useFlags, useHealth } from '../lib/queries'
import { formatCount } from '../lib/format'
import { Badge, Banner, Card, EmptyState, Skeleton, cx } from '../components/ui'
import { FlagRow, FlagRowHeader } from '../components/FlagRow'
import { StatTile } from '../components/StatTile'
import { FlagDetail } from './FlagDetail'

type StatusFilter = 'open' | 'reviewed' | 'all'

const STATUS_TABS: { id: StatusFilter; label: string }[] = [
  { id: 'open', label: 'Open' },
  { id: 'reviewed', label: 'Reviewed' },
  { id: 'all', label: 'All' },
]

export function FlagQueue() {
  const [status, setStatus] = useState<StatusFilter>('open')
  const [minScore, setMinScore] = useState(0)
  const [selected, setSelected] = useState<FlaggedTransaction | null>(null)

  const { data, isLoading, isError, error, isFetching } = useFlags(status, minScore)
  const { data: health } = useHealth()

  const flags = data?.flags ?? []
  const summary = data?.summary

  return (
    <div className="flex min-h-full flex-col">
      <header className="flex flex-wrap items-center justify-between gap-4 px-6 pt-6 pb-4">
        <div className="flex items-center gap-3">
          <h1 className="text-[20px] font-semibold tracking-tight text-[var(--ink)]">
            Flagged Transactions
          </h1>
          {summary && (
            <Badge
              icon="●"
              label={`${formatCount(summary.open_flags)} open`}
              colorVar="var(--accent)"
              softVar="var(--accent-soft)"
            />
          )}
          {isFetching && !isLoading && (
            <span className="text-[11px] text-[var(--ink-muted)]">refreshing…</span>
          )}
        </div>

        <div className="flex items-center gap-3">
          <ScoreFilter value={minScore} onChange={setMinScore} />
          <StatusTabs status={status} onChange={setStatus} />
        </div>
      </header>

      {/* Degraded-mode banner slot - hidden unless genuinely degraded (§4.2). */}
      <div className="px-6">
        <AnimatePresence>
          {health?.status === 'degraded' && (
            <div className="pb-4">
              <Banner title="Running in degraded mode">
                {health.components.database_reachable
                  ? 'Scoring is running without full graph context. Scores are still produced; some panels may be unavailable.'
                  : 'The audit database is unreachable, so new decisions may not be recorded. Check the Postgres container.'}
              </Banner>
            </div>
          )}
          {isError && (
            <div className="pb-4">
              <Banner tone="error" title="Can't load the flag queue">
                {messageFor(error)}
              </Banner>
            </div>
          )}
        </AnimatePresence>
      </div>

      {summary && (
        <div className="grid grid-cols-2 gap-3 px-6 pb-5 lg:grid-cols-4">
          <StatTile label="Open flags" value={summary.open_flags} hint="Awaiting review" />
          <StatTile
            label="Confirmed (24h)"
            value={summary.confirmed_today}
            accentVar="var(--risk-critical)"
            hint="Marked as fraud"
          />
          <StatTile
            label="Dismissed (24h)"
            value={summary.dismissed_today}
            accentVar="var(--risk-good)"
            hint="False positives"
          />
          <StatTile label="Serving model" value={summary.model_version} mono hint="Pinned version" />
        </div>
      )}

      <div className="min-h-0 flex-1 px-6 pb-6">
        <Card className="overflow-hidden">
          <FlagRowHeader />

          {isLoading ? (
            <QueueSkeleton />
          ) : flags.length === 0 ? (
            <QueueEmptyState status={status} minScore={minScore} />
          ) : (
            <div>
              {flags.map((flag, index) => (
                <FlagRow
                  key={flag.tx_id}
                  flag={flag}
                  index={index}
                  flagThreshold={0.9}
                  onReview={setSelected}
                  isSelected={selected?.tx_id === flag.tx_id}
                />
              ))}
              {data && data.count >= 200 && (
                <p className="px-5 py-3 text-[12px] text-[var(--ink-muted)]">
                  Showing the {formatCount(data.count)} highest-scoring flags. Raise the minimum
                  score to narrow the list.
                </p>
              )}
            </div>
          )}
        </Card>
      </div>

      <AnimatePresence>
        {selected && (
          <FlagDetail
            flag={selected}
            onClose={() => setSelected(null)}
            onDecided={() => setSelected(null)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

function StatusTabs({
  status,
  onChange,
}: {
  status: StatusFilter
  onChange: (status: StatusFilter) => void
}) {
  return (
    <div className="flex rounded-lg border border-[var(--border)] bg-[var(--surface)] p-0.5">
      {STATUS_TABS.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={cx(
            'relative rounded-md px-3 py-1.5 text-[12px] font-medium transition-colors',
            status === tab.id ? 'text-[var(--ink)]' : 'text-[var(--ink-secondary)]',
          )}
        >
          {status === tab.id && (
            <motion.span
              layoutId="status-tab"
              className="absolute inset-0 rounded-md bg-[var(--surface-hover)]"
              transition={{ type: 'spring', stiffness: 420, damping: 34 }}
            />
          )}
          <span className="relative">{tab.label}</span>
        </button>
      ))}
    </div>
  )
}

function ScoreFilter({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  return (
    <label className="flex items-center gap-2 text-[12px] text-[var(--ink-secondary)]">
      <span className="whitespace-nowrap">Min score</span>
      <input
        type="range"
        min={0}
        max={0.99}
        step={0.01}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1 w-28 cursor-pointer appearance-none rounded-full bg-[var(--border-strong)] accent-[var(--accent)]"
      />
      <span className="numeric w-9 text-[var(--ink)]">{value.toFixed(2)}</span>
    </label>
  )
}

function QueueSkeleton() {
  return (
    <div>
      {Array.from({ length: 6 }).map((_, index) => (
        <div
          key={index}
          className="grid grid-cols-[104px_minmax(120px,1fr)_minmax(200px,1.6fr)_minmax(160px,1.4fr)_132px_88px] items-center gap-4 border-b border-[var(--border)] px-5 py-4"
        >
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-4 w-24" />
          <Skeleton className="ml-auto h-7 w-16" />
        </div>
      ))}
    </div>
  )
}

/**
 * Empty is a *good* state here and is written as one. The wording also tells
 * the truth about this model's rate: at the pinned threshold roughly 4.7 in
 * 1,000 events are flagged, so a short replay legitimately produces nothing,
 * and an analyst should not read that as a broken pipeline.
 */
function QueueEmptyState({ status, minScore }: { status: StatusFilter; minScore: number }) {
  if (minScore > 0) {
    return (
      <EmptyState
        icon="◎"
        title="No flags above this score"
        description={`Nothing scored at or above ${minScore.toFixed(2)}. Lower the minimum score to see more of the queue.`}
      />
    )
  }
  if (status === 'reviewed') {
    return (
      <EmptyState
        icon="◷"
        title="Nothing reviewed yet"
        description="Decisions you make on open flags will appear here, with who made them and when."
      />
    )
  }
  return (
    <EmptyState
      title="No open flags"
      description="The stream is running and new flags will appear here automatically. Around 5 in every 1,000 transactions are flagged, so a short replay may produce none."
    />
  )
}
