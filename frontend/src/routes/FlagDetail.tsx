/**
 * Flag Detail (Design Doc §4.3) - the most important screen.
 *
 * Three stacked zones: verdict header, why-it-was-flagged, money-flow context,
 * then the two decisions. It opens as a panel over the queue so the analyst
 * keeps their place in the list.
 *
 * Failure is contained per panel: if Neo4j is down the graph panel says so and
 * the explanation above it still renders. One panel failing never blanks the
 * screen (§7).
 */

import { useEffect, useState } from 'react'
import { motion } from 'motion/react'
import type { AnalystDecision, FlaggedTransaction } from '../lib/api'
import { messageFor } from '../lib/api'
import { useGraph, useSubmitFeedback } from '../lib/queries'
import { decisionBadge, riskBand } from '../lib/risk'
import { formatAmount, formatDateTime, formatScore, MISSING } from '../lib/format'
import { Badge, Banner, Button, CardHeader, EmptyState, Field, Skeleton } from '../components/ui'
import { ShapFactors } from '../components/ShapFactors'
import { MoneyFlowGraph } from '../components/MoneyFlowGraph'

export function FlagDetail({
  flag,
  onClose,
  onDecided,
}: {
  flag: FlaggedTransaction
  onClose: () => void
  onDecided: () => void
}) {
  const band = riskBand(flag.score)
  const existingDecision = decisionBadge(flag.analyst_decision)
  const [confirming, setConfirming] = useState(false)
  const feedback = useSubmitFeedback()

  const graphQuery = useGraph(flag.account_key)

  // Escape closes the panel - a modal-ish surface that traps you is the fastest
  // way to make a keyboard user distrust the tool.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        if (confirming) setConfirming(false)
        else onClose()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, confirming])

  function decide(decision: AnalystDecision) {
    feedback.mutate(
      { txId: flag.tx_id, decision },
      {
        onSuccess: () => {
          setConfirming(false)
          onDecided()
        },
      },
    )
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.2 }}
        onClick={onClose}
        className="fixed inset-0 z-40 bg-black/45 backdrop-blur-[2px]"
        aria-hidden
      />

      <motion.aside
        role="dialog"
        aria-modal="true"
        aria-label={`Flag detail for transaction ${flag.tx_id}`}
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', stiffness: 320, damping: 36 }}
        className="fixed inset-y-0 right-0 z-50 flex w-full max-w-[620px] flex-col border-l border-[var(--border)] bg-[var(--plane)]"
      >
        {/* ---- Zone 1: verdict header ---- */}
        <header className="border-b border-[var(--border)] bg-[var(--surface)] px-5 py-4">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <Badge
                  icon={band.icon}
                  label={`${band.label} risk`}
                  colorVar={band.colorVar}
                  softVar={band.softVar}
                />
                <span className="numeric text-[13px] text-[var(--ink-secondary)]">
                  score {formatScore(flag.score)}
                </span>
                {existingDecision && (
                  <Badge
                    icon={existingDecision.icon}
                    label={existingDecision.label}
                    colorVar={existingDecision.colorVar}
                    softVar={existingDecision.softVar}
                  />
                )}
              </div>
              <p className="mt-2 text-[22px] leading-none font-semibold tracking-tight text-[var(--ink)]">
                {formatAmount(flag.amount_paid, flag.payment_currency)}
              </p>
              <p className="identifier mt-1.5 text-[11px] text-[var(--ink-muted)]">{flag.tx_id}</p>
            </div>

            <Button variant="ghost" onClick={onClose} aria-label="Close">
              ✕
            </Button>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Field label="Sender" value={flag.account_key} mono title={flag.account_key} />
            <Field
              label="Receiver"
              value={flag.receiver_account_key ?? MISSING}
              mono
              title={flag.receiver_account_key ?? undefined}
            />
            <Field label="Method" value={flag.payment_format ?? MISSING} />
            <Field label="Transaction time" value={formatDateTime(flag.tx_timestamp)} />
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {/* ---- Zone 2: why this was flagged ---- */}
          <section className="border-b border-[var(--border)] bg-[var(--surface)]">
            <CardHeader
              title="Why this was flagged"
              description="The factors that moved this transaction's score the most."
            />
            <ShapFactors factors={flag.explanation} />
          </section>

          {/* ---- Zone 3: money-flow context ---- */}
          <section className="bg-[var(--surface)]">
            <CardHeader
              title="Money-flow context"
              description={`Accounts within two hops of ${flag.account_key}.`}
            />
            <div className="px-5 py-4">
              {graphQuery.isLoading ? (
                <div className="space-y-3">
                  <Skeleton className="h-[280px] w-full" />
                  <p className="text-center text-[12px] text-[var(--ink-muted)]">
                    Building money-flow view…
                  </p>
                </div>
              ) : graphQuery.isError ? (
                // Honest and contained: the panel fails, the screen does not.
                <Banner tone="warning" title="Graph view unavailable">
                  {messageFor(graphQuery.error)} The explanation above is unaffected — it comes
                  from the scoring service, which does not read the graph store.
                </Banner>
              ) : graphQuery.data && graphQuery.data.nodes.length > 0 ? (
                <MoneyFlowGraph graph={graphQuery.data} focusAccount={flag.account_key} />
              ) : (
                <EmptyState
                  icon="◌"
                  title="No connected accounts found"
                  description="This account has no neighbours in the graph store for the loaded window."
                />
              )}
            </div>
          </section>
        </div>

        {/* ---- Decisions ---- */}
        <footer className="border-t border-[var(--border)] bg-[var(--surface)] px-5 py-4">
          {feedback.isError && (
            <div className="mb-3">
              <Banner tone="error" title="Could not record that decision">
                {messageFor(feedback.error)}
              </Banner>
            </div>
          )}

          {existingDecision ? (
            <div className="flex items-center justify-between gap-3">
              <p className="text-[13px] text-[var(--ink-secondary)]">
                Reviewed by{' '}
                <span className="font-medium text-[var(--ink)]">{flag.decided_by}</span> on{' '}
                {formatDateTime(flag.decided_at)}.
              </p>
              <Button variant="secondary" onClick={onClose}>
                Close
              </Button>
            </div>
          ) : confirming ? (
            // Confirm-fraud is consequential, so it takes two clicks. Dismiss
            // stays one - it is reversible and visible in the audit log.
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex flex-wrap items-center justify-between gap-3"
            >
              <p className="text-[13px] text-[var(--ink)]">
                Mark <span className="identifier">{flag.tx_id}</span> as confirmed fraud?
              </p>
              <div className="flex gap-2">
                <Button variant="ghost" onClick={() => setConfirming(false)}>
                  Cancel
                </Button>
                <Button
                  variant="danger"
                  loading={feedback.isPending}
                  onClick={() => decide('confirmed_fraud')}
                >
                  Yes, confirm fraud
                </Button>
              </div>
            </motion.div>
          ) : (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="text-[12px] text-[var(--ink-muted)]">
                Your decision is recorded in the audit log and feeds the next retrain.
              </p>
              <div className="flex gap-2">
                <Button
                  variant="secondary"
                  loading={feedback.isPending}
                  onClick={() => decide('false_positive')}
                >
                  Dismiss
                </Button>
                <Button variant="danger" onClick={() => setConfirming(true)}>
                  Confirm fraud
                </Button>
              </div>
            </div>
          )}
        </footer>
      </motion.aside>
    </>
  )
}
