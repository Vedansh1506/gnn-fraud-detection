/**
 * "Why this was flagged" (Design Doc §4.3, Principle 2).
 *
 * A diverging bar per factor: bars grow right from a centre baseline when the
 * factor raised the score and left when it lowered it. Diverging is the right
 * form because the data has a natural zero and a sign that means something -
 * a plain ranked bar chart of absolute values would throw away the direction,
 * which is the most useful part for an analyst.
 *
 * The plain-language sentence is the headline and the feature name is the
 * footnote, not the other way round. The SHAP value is shown too - hiding the
 * number would be a different kind of dishonesty - but it is never the only
 * thing on the row.
 */

import { motion } from 'motion/react'
import type { ExplanationFactor } from '../lib/api'
import { contributionDirection } from '../lib/risk'

export function ShapFactors({ factors }: { factors: ExplanationFactor[] }) {
  if (factors.length === 0) {
    return (
      <p className="px-5 py-6 text-[13px] text-[var(--ink-secondary)]">
        No explanation was recorded for this transaction.
      </p>
    )
  }

  // Scale to the largest absolute contribution so the bars use the full width;
  // the axis is shared across rows, so lengths remain comparable to each other.
  const scale = Math.max(...factors.map((factor) => Math.abs(factor.contribution)))

  return (
    <div className="px-5 py-4">
      <div className="space-y-4">
        {factors.map((factor, index) => (
          <FactorRow key={factor.feature} factor={factor} scale={scale} index={index} />
        ))}
      </div>

      <p className="mt-5 border-t border-[var(--border)] pt-3 text-[11px] leading-relaxed text-[var(--ink-muted)]">
        Contributions are SHAP values from the serving model: how much each factor moved this
        transaction's score, relative to an average transaction. They explain the model's
        reasoning — they are not evidence of intent.
      </p>
    </div>
  )
}

function FactorRow({
  factor,
  scale,
  index,
}: {
  factor: ExplanationFactor
  scale: number
  index: number
}) {
  const direction = contributionDirection(factor.contribution)
  const magnitude = scale > 0 ? Math.abs(factor.contribution) / scale : 0
  const raises = factor.contribution >= 0

  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <p className="text-[13px] leading-snug text-[var(--ink)]">{factor.plain}</p>
        <span
          className="numeric shrink-0 text-[12px] font-medium"
          style={{ color: direction.colorVar }}
        >
          {factor.contribution >= 0 ? '+' : '−'}
          {Math.abs(factor.contribution).toFixed(2)}
        </span>
      </div>

      {/* Centre baseline: left of it lowers risk, right of it raises it. */}
      <div className="relative h-2.5 w-full">
        <div
          className="absolute inset-y-0 left-1/2 w-px"
          style={{ background: 'var(--border-strong)' }}
          aria-hidden
        />
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${magnitude * 50}%` }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1], delay: 0.06 + index * 0.05 }}
          className="absolute inset-y-0"
          style={{
            left: raises ? '50%' : undefined,
            right: raises ? undefined : '50%',
            background: direction.colorVar,
            // Rounded only on the growing end, anchored square to the baseline.
            borderRadius: raises ? '0 4px 4px 0' : '4px 0 0 4px',
          }}
        />
      </div>

      <p className="mt-1.5 flex items-center gap-2 text-[11px] text-[var(--ink-muted)]">
        <span style={{ color: direction.colorVar }}>{direction.label}</span>
        <span aria-hidden>·</span>
        <span className="identifier">{factor.feature}</span>
      </p>
    </div>
  )
}
