/**
 * Baseline vs GNN on AUPRC (Design Doc §4.4) - the project's headline claim.
 *
 * Two bars, one axis, direct-labelled. Two bars of the same measure is a
 * magnitude comparison, so bar length is the encoding and the axis starts at
 * zero; a truncated axis here would inflate a genuinely modest +10% into
 * something that looks like a breakthrough, which is the exact dishonesty this
 * project's own rules forbid.
 *
 * Each bar is direct-labelled, so colour never has to carry identity.
 */

import { motion } from 'motion/react'
import { formatMetric } from '../lib/format'

export function LiftChart({
  baseline,
  gnn,
  baselineLabel,
  gnnLabel,
}: {
  baseline: number
  gnn: number
  baselineLabel: string
  gnnLabel: string
}) {
  // A little headroom so the longer bar doesn't touch the container edge.
  const scale = Math.max(baseline, gnn) * 1.18

  return (
    <div className="space-y-4">
      <Bar
        label={baselineLabel}
        sublabel="Tabular only"
        value={baseline}
        fraction={baseline / scale}
        color="var(--viz-series-1)"
        delay={0.05}
      />
      <Bar
        label={gnnLabel}
        sublabel="With GraphSAGE embeddings"
        value={gnn}
        fraction={gnn / scale}
        color="var(--viz-series-2)"
        delay={0.18}
      />
      <p className="border-t border-[var(--border)] pt-3 text-[11px] leading-relaxed text-[var(--ink-muted)]">
        AUPRC on the held-out test window, same split and same evaluation for both — the only
        difference is the 64 embedding columns. Axis starts at zero.
      </p>
    </div>
  )
}

function Bar({
  label,
  sublabel,
  value,
  fraction,
  color,
  delay,
}: {
  label: string
  sublabel: string
  value: number
  fraction: number
  color: string
  delay: number
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <div className="min-w-0">
          <span className="identifier text-[12px] text-[var(--ink)]">{label}</span>
          <span className="ml-2 text-[11px] text-[var(--ink-muted)]">{sublabel}</span>
        </div>
        <span className="numeric text-[13px] font-semibold text-[var(--ink)]">
          {formatMetric(value)}
        </span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-sm bg-[var(--viz-grid)]">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.max(fraction, 0) * 100}%` }}
          transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1], delay }}
          className="h-full"
          style={{ background: color, borderRadius: '0 4px 4px 0' }}
        />
      </div>
    </div>
  )
}
