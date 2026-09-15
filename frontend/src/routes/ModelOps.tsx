/**
 * Model & Ops (Design Doc §4.4) - the operator's screen.
 *
 * Built around the one number this whole project exists to produce: the
 * baseline-vs-GNN AUPRC lift. Everything here is read from the pinned artifacts
 * and the feedback table; nothing is computed in the browser and nothing is
 * filled in with a plausible-looking placeholder.
 *
 * Two honesty rules are enforced in the markup rather than left to whoever
 * writes the demo script:
 *   - the ROC-AUC drop is shown next to the AUPRC gain, because quoting the
 *     gain alone is cherry-picking;
 *   - drift renders as "not measured" rather than green, because nothing has
 *     measured it yet.
 */

import { useEffect, useRef, useState } from 'react'
import Lenis from 'lenis'
import { motion } from 'motion/react'
import { useAuth } from '../lib/auth'
import { useModels, useTriggerRetrain } from '../lib/queries'
import { messageFor, type ModelVersion } from '../lib/api'
import { formatCount, formatMetric, formatPercent, formatRelative, MISSING } from '../lib/format'
import { formatDateTime } from '../lib/format'
import { Badge, Banner, Button, Card, CardHeader, Skeleton } from '../components/ui'
import { cx } from '../lib/cx'
import { CountUp, StatTile } from '../components/StatTile'
import { LiftChart } from '../components/LiftChart'

/** Below this many reviews, an override rate is noise rather than a signal. */
const SAMPLE_FLOOR = 20

export function ModelOps() {
  const { isOperator } = useAuth()
  const { data, isLoading, isError, error } = useModels()
  const scrollRef = useRef<HTMLDivElement>(null)

  // Smooth scrolling on this screen only. It is the one long, read-through
  // page in the app; the queue is a worklist where native scrolling and
  // keyboard paging matter more than feel.
  useEffect(() => {
    const wrapper = scrollRef.current
    if (!wrapper) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return

    const lenis = new Lenis({ wrapper, content: wrapper.firstElementChild as HTMLElement })
    let frame = 0
    const raf = (time: number) => {
      lenis.raf(time)
      frame = requestAnimationFrame(raf)
    }
    frame = requestAnimationFrame(raf)

    return () => {
      cancelAnimationFrame(frame)
      lenis.destroy()
    }
  }, [])

  if (isLoading) return <OpsSkeleton />

  if (isError || !data) {
    return (
      <div className="p-6">
        <Banner tone="error" title="Can't load model information">
          {messageFor(error)}
        </Banner>
      </div>
    )
  }

  const hasComparison = data.baseline_auprc !== null && data.gnn_auprc !== null
  const baseline = data.versions.find((version) => version.embedding_version === null)
  const gnn = data.versions.find((version) => version.embedding_version !== null)

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto">
      <div className="mx-auto max-w-[1000px] px-6 py-6">
        <header className="mb-6">
          <h1 className="text-[20px] font-semibold tracking-tight text-[var(--ink)]">
            Model &amp; Ops
          </h1>
          <p className="mt-1 text-[13px] text-[var(--ink-secondary)]">
            Serving{' '}
            <span className="identifier text-[var(--ink)]">{data.serving_model_version}</span> with
            embeddings{' '}
            <span className="identifier text-[var(--ink)]">{data.serving_embedding_version}</span>,
            flagging at score ≥ {data.flag_threshold}.
          </p>
        </header>

        {hasComparison && (
          <HeadlineLift
            liftPct={data.auprc_lift_pct}
            baseline={data.baseline_auprc as number}
            gnn={data.gnn_auprc as number}
            baselineVersion={baseline}
            gnnVersion={gnn}
          />
        )}

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader
              title="Baseline vs GNN"
              description="Average precision on the held-out window."
            />
            <div className="px-5 py-4">
              {hasComparison ? (
                <LiftChart
                  baseline={data.baseline_auprc as number}
                  gnn={data.gnn_auprc as number}
                  baselineLabel={baseline?.version ?? 'baseline'}
                  gnnLabel={gnn?.version ?? 'gnn'}
                />
              ) : (
                <p className="text-[13px] text-[var(--ink-secondary)]">
                  Needs one tabular and one embedding-fused version to compare. Train both to see
                  the lift.
                </p>
              )}
            </div>
          </Card>

          <Card>
            <CardHeader
              title="Analyst agreement"
              description="How often reviewers overturn the model."
            />
            <div className="px-5 py-4">
              {data.reviewed_count === 0 ? (
                <p className="text-[13px] text-[var(--ink-secondary)]">
                  No flags reviewed yet. Once analysts start deciding, the override rate appears
                  here — it is the only live quality signal this system has between retrains.
                </p>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <StatTile
                      label="Override rate"
                      value={formatPercent(data.override_rate)}
                      // Deliberately NOT colour-coded green/good. This project
                      // has never established what an acceptable override rate
                      // is, so painting a number green would assert a judgement
                      // nothing has earned. Amber only past a half, where "the
                      // analyst rejects most of what the model raises" is worth
                      // a second look regardless of where the bar eventually sits.
                      accentVar={
                        (data.override_rate ?? 0) > 0.5 ? 'var(--risk-warning)' : 'var(--ink)'
                      }
                      hint="Flags dismissed as false positives"
                    />
                    <StatTile
                      label="Reviewed"
                      value={data.reviewed_count}
                      hint="Decisions recorded"
                    />
                  </div>
                  {data.reviewed_count < SAMPLE_FLOOR && (
                    <p className="mt-3 text-[11px] leading-relaxed text-[var(--ink-muted)]">
                      Based on {data.reviewed_count} decision
                      {data.reviewed_count === 1 ? '' : 's'} — too few to read a trend into.
                    </p>
                  )}
                </>
              )}
            </div>
          </Card>
        </div>

        <div className="mt-5">
          <Card>
            <CardHeader
              title="Trained versions"
              description="Read from each version's metrics.json — every number was measured by an evaluation run."
            />
            <VersionTable versions={data.versions} />
          </Card>
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-2">
          <DriftPanel drift={data.drift} />
          <RetrainPanel
            isOperator={isOperator}
            recent={data.recent_retrains}
            pendingHint={data.reviewed_count}
          />
        </div>
      </div>
    </div>
  )
}

/**
 * The hero. The lift is the headline, but the ROC-AUC drop sits immediately
 * beside it - not in a footnote - because the honest version of this result is
 * "better at ranking the top of the queue, worse at global ordering".
 */
function HeadlineLift({
  liftPct,
  baseline,
  gnn,
  baselineVersion,
  gnnVersion,
}: {
  liftPct: number | null
  baseline: number
  gnn: number
  baselineVersion?: ModelVersion
  gnnVersion?: ModelVersion
}) {
  const rocDelta =
    baselineVersion && gnnVersion ? gnnVersion.roc_auc - baselineVersion.roc_auc : null
  const recallDelta =
    baselineVersion && gnnVersion
      ? gnnVersion.best_f1_recall - baselineVersion.best_f1_recall
      : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
      className="relative overflow-hidden rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--surface)] px-6 py-6"
      style={{ boxShadow: 'var(--shadow-card)' }}
    >
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.06]"
        style={{
          background:
            'radial-gradient(ellipse 60% 120% at 12% 0%, var(--accent) 0%, transparent 60%)',
        }}
        aria-hidden
      />

      <div className="relative flex flex-wrap items-end justify-between gap-6">
        <div>
          <p className="text-[11px] font-medium tracking-wide text-[var(--ink-muted)] uppercase">
            Graph lift on AUPRC
          </p>
          <p className="mt-2 flex items-baseline gap-1.5 text-[54px] leading-none font-semibold tracking-tight text-[var(--ink)]">
            <span style={{ color: 'var(--risk-good)' }}>
              +<CountUp value={liftPct ?? 0} decimals={1} />%
            </span>
          </p>
          <p className="mt-2.5 text-[13px] text-[var(--ink-secondary)]">
            {formatMetric(baseline)} → {formatMetric(gnn)} average precision
          </p>
        </div>

        <div className="flex flex-wrap gap-6">
          {recallDelta !== null && (
            <DeltaStat
              label="Recall at best F1"
              value={`${formatPercent(baselineVersion!.best_f1_recall, 1)} → ${formatPercent(gnnVersion!.best_f1_recall, 1)}`}
              tone="good"
              note="The practically useful gain"
            />
          )}
          {rocDelta !== null && (
            <DeltaStat
              label="ROC-AUC"
              value={`${formatMetric(baselineVersion!.roc_auc, 3)} → ${formatMetric(gnnVersion!.roc_auc, 3)}`}
              tone={rocDelta < 0 ? 'bad' : 'good'}
              note={rocDelta < 0 ? 'Down — reported anyway' : 'Up'}
            />
          )}
        </div>
      </div>

      <p className="relative mt-5 border-t border-[var(--border)] pt-3 text-[11px] leading-relaxed text-[var(--ink-muted)]">
        The baseline already includes hand-built graph aggregates, so this measures the GNN's
        marginal value over those — not "graph versus no graph". The lift is real but modest in
        absolute terms.
      </p>
    </motion.div>
  )
}

function DeltaStat({
  label,
  value,
  tone,
  note,
}: {
  label: string
  value: string
  tone: 'good' | 'bad'
  note: string
}) {
  const color = tone === 'good' ? 'var(--risk-good)' : 'var(--risk-critical)'
  return (
    <div>
      <p className="text-[11px] font-medium tracking-wide text-[var(--ink-muted)] uppercase">
        {label}
      </p>
      <p className="numeric mt-1.5 text-[16px] font-semibold text-[var(--ink)]">{value}</p>
      <p className="mt-0.5 text-[11px]" style={{ color }}>
        {tone === 'good' ? '↑' : '↓'} {note}
      </p>
    </div>
  )
}

function VersionTable({ versions }: { versions: ModelVersion[] }) {
  if (versions.length === 0) {
    return (
      <p className="px-5 py-6 text-[13px] text-[var(--ink-secondary)]">
        No trained versions found in the artifact directory.
      </p>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[720px] text-left">
        <thead>
          <tr className="border-b border-[var(--border)] text-[11px] tracking-wide text-[var(--ink-muted)] uppercase">
            <th className="px-5 py-2.5 font-medium">Version</th>
            <th className="px-5 py-2.5 font-medium">Embeddings</th>
            <th className="px-5 py-2.5 text-right font-medium">AUPRC</th>
            <th className="px-5 py-2.5 text-right font-medium">ROC-AUC</th>
            <th className="px-5 py-2.5 text-right font-medium">Recall @ best F1</th>
            <th className="px-5 py-2.5 text-right font-medium">Test rows</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((version) => (
            <tr
              key={version.version}
              className={cx(
                'border-b border-[var(--border)] text-[13px] last:border-0',
                version.is_serving && 'bg-[var(--accent-soft)]',
              )}
            >
              <td className="px-5 py-3">
                <div className="flex items-center gap-2">
                  <span className="identifier text-[var(--ink)]">{version.version}</span>
                  {version.is_serving && (
                    <Badge
                      icon="●"
                      label="Serving"
                      colorVar="var(--accent)"
                      softVar="var(--accent-soft)"
                    />
                  )}
                </div>
              </td>
              <td className="identifier px-5 py-3 text-[12px] text-[var(--ink-secondary)]">
                {version.embedding_version ?? MISSING}
              </td>
              <td className="numeric px-5 py-3 text-right font-medium text-[var(--ink)]">
                {formatMetric(version.auprc)}
              </td>
              <td className="numeric px-5 py-3 text-right text-[var(--ink-secondary)]">
                {formatMetric(version.roc_auc, 3)}
              </td>
              <td className="numeric px-5 py-3 text-right text-[var(--ink-secondary)]">
                {formatPercent(version.best_f1_recall)}
              </td>
              <td className="numeric px-5 py-3 text-right text-[var(--ink-secondary)]">
                {formatCount(version.test_rows)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/**
 * Each drift state gets its own treatment. Before real states existed this
 * painted everything that wasn't "not measured" green - which would have shown
 * an `alert` as healthy, the precise failure this panel exists to prevent.
 */
const DRIFT_STATES: Record<
  string,
  { icon: string; label: string; colorVar: string; softVar: string }
> = {
  ok: {
    icon: '●',
    label: 'No drift detected',
    colorVar: 'var(--risk-good)',
    softVar: 'var(--risk-good-soft)',
  },
  warning: {
    icon: '▲',
    label: 'Some drift',
    colorVar: 'var(--risk-warning)',
    softVar: 'var(--risk-warning-soft)',
  },
  alert: {
    icon: '!',
    label: 'Significant drift',
    colorVar: 'var(--risk-critical)',
    softVar: 'var(--risk-critical-soft)',
  },
  not_instrumented: {
    icon: '○',
    label: 'Not measured',
    colorVar: 'var(--ink-muted)',
    softVar: 'transparent',
  },
}

function DriftPanel({
  drift,
}: {
  drift: { state: string; message: string; checked_at: string | null }
}) {
  const tone = DRIFT_STATES[drift.state] ?? DRIFT_STATES.not_instrumented

  return (
    <Card>
      <CardHeader
        title="Feature drift"
        description="Are live inputs still like the data the model was validated on?"
      />
      <div className="px-5 py-4">
        <Badge
          icon={tone.icon}
          label={tone.label}
          colorVar={tone.colorVar}
          softVar={tone.softVar}
        />
        <p className="mt-3 text-[13px] leading-relaxed text-[var(--ink-secondary)]">
          {drift.message}
        </p>
        {/* Absent when nothing was computed - and absent is the honest answer
            there, rather than a timestamp implying a check that never ran. */}
        {drift.checked_at && (
          <p className="mt-2 text-[11px] text-[var(--ink-muted)]">
            Checked {formatRelative(drift.checked_at)}
          </p>
        )}
      </div>
    </Card>
  )
}

function RetrainPanel({
  isOperator,
  recent,
  pendingHint,
}: {
  isOperator: boolean
  recent: {
    job_id: string
    status: string
    requested_by: string
    requested_at: string
    resulting_model_version: string | null
  }[]
  pendingHint: number
}) {
  const [confirming, setConfirming] = useState(false)
  const retrain = useTriggerRetrain()

  return (
    <Card>
      <CardHeader
        title="Retrain"
        description="Records a retrain request. Training itself runs offline on GPU."
      />
      <div className="px-5 py-4">
        {!isOperator ? (
          <p className="text-[13px] text-[var(--ink-secondary)]">
            Requesting a retrain needs the operator role. You're signed in as an analyst.
          </p>
        ) : retrain.isSuccess ? (
          <Banner tone="info" title={`Requested — ${retrain.data.job_id}`}>
            <p className="mb-2">
              {retrain.data.feedback_rows_pending} unused feedback row
              {retrain.data.feedback_rows_pending === 1 ? '' : 's'} at request time. The API has no
              GPU, so run the steps below and promote only after the eval gate passes:
            </p>
            <pre className="identifier overflow-x-auto rounded-md bg-[var(--plane)] p-2.5 text-[11px] leading-relaxed whitespace-pre-wrap text-[var(--ink-secondary)]">
              {retrain.data.instructions}
            </pre>
          </Banner>
        ) : confirming ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-[13px] text-[var(--ink)]">Record a retrain request?</p>
            <div className="flex gap-2">
              <Button variant="ghost" onClick={() => setConfirming(false)}>
                Cancel
              </Button>
              <Button
                variant="primary"
                loading={retrain.isPending}
                onClick={() => retrain.mutate()}
              >
                Yes, request retrain
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-[12px] text-[var(--ink-muted)]">
              {pendingHint > 0
                ? `${pendingHint} analyst decision${pendingHint === 1 ? '' : 's'} recorded so far.`
                : 'No analyst feedback recorded yet.'}
            </p>
            <Button variant="primary" onClick={() => setConfirming(true)}>
              Trigger retrain
            </Button>
          </div>
        )}

        {retrain.isError && (
          <div className="mt-3">
            <Banner tone="error" title="Retrain request failed">
              {messageFor(retrain.error)}
            </Banner>
          </div>
        )}

        {recent.length > 0 && (
          <div className="mt-4 border-t border-[var(--border)] pt-3">
            <p className="mb-2 text-[11px] font-medium tracking-wide text-[var(--ink-muted)] uppercase">
              Recent requests
            </p>
            <ul className="space-y-2">
              {recent.map((run) => (
                <li key={run.job_id} className="flex items-center justify-between gap-3 text-[12px]">
                  <span className="flex min-w-0 items-center gap-2">
                    <RetrainStatus status={run.status} />
                    <span className="identifier truncate text-[var(--ink-secondary)]">
                      {run.resulting_model_version ?? run.job_id}
                    </span>
                  </span>
                  <span className="shrink-0 text-[var(--ink-muted)]">
                    {run.requested_by} · {formatDateTime(run.requested_at)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  )
}

/**
 * An open request and a finished one are different facts, and the list showed
 * them identically. "Requested" deliberately reads as neutral rather than
 * positive - nothing has happened yet, and the API cannot make it happen.
 */
function RetrainStatus({ status }: { status: string }) {
  const tone =
    status === 'completed'
      ? { icon: '✓', colorVar: 'var(--risk-good)', softVar: 'var(--risk-good-soft)' }
      : status === 'failed'
        ? { icon: '!', colorVar: 'var(--risk-critical)', softVar: 'var(--risk-critical-soft)' }
        : status === 'running'
          ? { icon: '◐', colorVar: 'var(--accent)', softVar: 'var(--accent-soft)' }
          : { icon: '○', colorVar: 'var(--ink-muted)', softVar: 'transparent' }

  return (
    <Badge
      icon={tone.icon}
      label={status}
      colorVar={tone.colorVar}
      softVar={tone.softVar}
      className="shrink-0 capitalize"
    />
  )
}

function OpsSkeleton() {
  return (
    <div className="mx-auto max-w-[1000px] space-y-5 px-6 py-6">
      <Skeleton className="h-8 w-48" />
      <Skeleton className="h-[168px] w-full rounded-[var(--radius-card)]" />
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="h-[220px] w-full rounded-[var(--radius-card)]" />
        <Skeleton className="h-[220px] w-full rounded-[var(--radius-card)]" />
      </div>
      <Skeleton className="h-[240px] w-full rounded-[var(--radius-card)]" />
    </div>
  )
}
