/**
 * The summary strip's stat tiles (Design Doc §4.2).
 *
 * A count is a headline, not a chart - no sparkline, no bar, nothing to decode.
 * The number animates to its new value rather than snapping, so an analyst
 * glancing away and back can tell that something changed.
 */

import { animate, useMotionValue, useTransform, motion } from 'motion/react'
import { useEffect } from 'react'
import { cx } from './ui'

export function StatTile({
  label,
  value,
  hint,
  accentVar,
  mono = false,
}: {
  label: string
  value: number | string
  hint?: string
  accentVar?: string
  mono?: boolean
}) {
  const isNumeric = typeof value === 'number'

  return (
    <div className="flex flex-col justify-between rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--surface)] px-4 py-3.5">
      <p className="text-[11px] font-medium tracking-wide text-[var(--ink-muted)] uppercase">
        {label}
      </p>
      <p
        className={cx(
          'mt-2 text-[26px] leading-none font-semibold tracking-tight',
          mono && 'identifier text-[15px]',
        )}
        style={{ color: accentVar ?? 'var(--ink)' }}
      >
        {isNumeric ? <CountUp value={value} /> : value}
      </p>
      {hint && <p className="mt-1.5 text-[11px] text-[var(--ink-muted)]">{hint}</p>}
    </div>
  )
}

/** Counts up to `value`; respects reduced-motion by snapping instead. */
export function CountUp({ value, decimals = 0 }: { value: number; decimals?: number }) {
  const motionValue = useMotionValue(0)
  const text = useTransform(motionValue, (latest) =>
    latest.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    }),
  )

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      motionValue.set(value)
      return
    }
    const controls = animate(motionValue, value, { duration: 0.7, ease: [0.22, 1, 0.36, 1] })
    return () => controls.stop()
  }, [value, motionValue])

  return <motion.span className="numeric">{text}</motion.span>
}
