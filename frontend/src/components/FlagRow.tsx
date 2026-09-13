/**
 * One flag row - the single definition used by the queue (Design Doc §4.5).
 *
 * Column widths are fixed across every row so the queue reads as a table and
 * can be scanned vertically; ragged rows are the fastest way to make a worklist
 * unscannable.
 */

import { motion } from 'motion/react'
import type { FlaggedTransaction } from '../lib/api'
import { decisionBadge, riskBand } from '../lib/risk'
import {
  MISSING,
  formatAmount,
  formatDateTime,
  formatRelative,
  formatScore,
  shortAccount,
} from '../lib/format'
import { Badge, Button, cx } from './ui'

export function FlagRow({
  flag,
  index,
  flagThreshold,
  onReview,
  isSelected,
}: {
  flag: FlaggedTransaction
  index: number
  flagThreshold: number
  onReview: (flag: FlaggedTransaction) => void
  isSelected: boolean
}) {
  const band = riskBand(flag.score, flagThreshold)
  const decision = decisionBadge(flag.analyst_decision)
  const topFactor = flag.explanation[0]

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{
        duration: 0.34,
        ease: [0.22, 1, 0.36, 1],
        // Staggered, but capped: past ~12 rows the delay stops meaning
        // anything and just makes the list feel slow to arrive.
        delay: Math.min(index, 12) * 0.022,
      }}
      onClick={() => onReview(flag)}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          onReview(flag)
        }
      }}
      className={cx(
        'group grid cursor-pointer grid-cols-[104px_minmax(120px,1fr)_minmax(200px,1.6fr)_minmax(160px,1.4fr)_132px_88px]',
        'items-center gap-4 border-b border-[var(--border)] px-5 py-3',
        'transition-colors duration-[var(--duration-fast)] hover:bg-[var(--surface-hover)]',
        isSelected && 'bg-[var(--accent-soft)]',
      )}
    >
      {/* Risk: colour + icon + word, never colour alone. */}
      <div className="flex flex-col items-start gap-1">
        <Badge
          icon={band.icon}
          label={band.label}
          colorVar={band.colorVar}
          softVar={band.softVar}
        />
        <span className="numeric pl-1 text-[11px] text-[var(--ink-muted)]">
          {formatScore(flag.score)}
        </span>
      </div>

      <div className="numeric truncate text-[13px] font-medium text-[var(--ink)]">
        {formatAmount(flag.amount_paid, flag.payment_currency)}
      </div>

      {/* Sender → receiver, the thing an analyst actually triages on. */}
      <div className="flex min-w-0 items-center gap-2 text-[12px]">
        <span
          className="identifier truncate text-[var(--ink-secondary)]"
          title={flag.account_key}
        >
          {shortAccount(flag.account_key)}
        </span>
        <span className="shrink-0 text-[var(--ink-muted)]" aria-label="pays">
          →
        </span>
        <span
          className="identifier truncate text-[var(--ink-secondary)]"
          title={flag.receiver_account_key ?? undefined}
        >
          {shortAccount(flag.receiver_account_key)}
        </span>
      </div>

      {/* The plain-language reason, not the feature name (Principle 2). */}
      <div className="min-w-0 truncate text-[12px] text-[var(--ink-secondary)]">
        {topFactor ? topFactor.plain : MISSING}
      </div>

      <div className="text-[12px] text-[var(--ink-muted)]">
        <span title={formatDateTime(flag.tx_timestamp)}>
          {flag.tx_timestamp ? formatDateTime(flag.tx_timestamp) : MISSING}
        </span>
        <span className="block text-[11px] opacity-70" title="When this was scored">
          scored {formatRelative(flag.scored_at)}
        </span>
      </div>

      <div className="flex justify-end">
        {decision ? (
          <Badge
            icon={decision.icon}
            label={decision.label}
            colorVar={decision.colorVar}
            softVar={decision.softVar}
          />
        ) : (
          <Button
            variant="secondary"
            className="opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
            onClick={(event) => {
              event.stopPropagation()
              onReview(flag)
            }}
          >
            Review
          </Button>
        )}
      </div>
    </motion.div>
  )
}

export function FlagRowHeader() {
  return (
    <div
      className={cx(
        'grid grid-cols-[104px_minmax(120px,1fr)_minmax(200px,1.6fr)_minmax(160px,1.4fr)_132px_88px]',
        'items-center gap-4 border-b border-[var(--border)] px-5 py-2.5',
        'text-[11px] font-medium tracking-wide text-[var(--ink-muted)] uppercase',
      )}
    >
      <span>Risk</span>
      <span>Amount</span>
      <span>Sender → Receiver</span>
      <span>Top reason</span>
      <span>Transaction time</span>
      <span className="text-right">Action</span>
    </div>
  )
}
