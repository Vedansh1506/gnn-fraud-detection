/**
 * The shared primitives. Every screen composes these rather than styling its
 * own boxes, which is what keeps the queue, the detail view and the ops screen
 * looking like one product (Design Doc §4.5 component rule).
 */

import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { cx } from '../lib/cx'

/* ---------------------------------------------------------------- surfaces */

export function Card({
  children,
  className,
  ...rest
}: { children: ReactNode; className?: string } & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cx(
        'rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--surface)]',
        className,
      )}
      style={{ boxShadow: 'var(--shadow-card)' }}
      {...rest}
    >
      {children}
    </div>
  )
}

export function CardHeader({
  title,
  description,
  actions,
}: {
  title: ReactNode
  description?: ReactNode
  actions?: ReactNode
}) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-[var(--border)] px-5 py-4">
      <div className="min-w-0">
        <h2 className="text-[13px] font-semibold tracking-wide text-[var(--ink)] uppercase">
          {title}
        </h2>
        {description && (
          <p className="mt-1 text-[13px] leading-relaxed text-[var(--ink-secondary)]">
            {description}
          </p>
        )}
      </div>
      {actions && <div className="shrink-0">{actions}</div>}
    </div>
  )
}

/* ----------------------------------------------------------------- buttons */

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  // Exactly one loud button per screen (Principle 3); everything else recedes.
  primary:
    'bg-[var(--accent)] text-[var(--accent-ink)] hover:bg-[var(--accent-hover)] border-transparent',
  secondary:
    'bg-[var(--surface-raised)] text-[var(--ink)] hover:bg-[var(--surface-hover)] border-[var(--border-strong)]',
  ghost:
    'bg-transparent text-[var(--ink-secondary)] hover:text-[var(--ink)] hover:bg-[var(--surface-hover)] border-transparent',
  danger:
    'bg-[var(--risk-critical)] text-white hover:brightness-110 border-transparent',
}

export function Button({
  variant = 'secondary',
  className,
  loading = false,
  children,
  ...rest
}: {
  variant?: ButtonVariant
  loading?: boolean
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-lg border px-3.5 py-2',
        'text-[13px] font-medium whitespace-nowrap',
        'transition-[background-color,color,filter,opacity] duration-[var(--duration-fast)]',
        'disabled:cursor-not-allowed disabled:opacity-50',
        BUTTON_STYLES[variant],
        className,
      )}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading && <Spinner />}
      {children}
    </button>
  )
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      className={cx(
        'inline-block h-3.5 w-3.5 shrink-0 animate-spin rounded-full',
        'border-2 border-current border-t-transparent opacity-70',
        className,
      )}
      aria-hidden
    />
  )
}

/* ------------------------------------------------------------------ badges */

/**
 * The single badge definition. Colour is never the only channel - an icon and
 * a word always travel with it, which is what makes the queue readable with
 * red/green colour blindness (§11).
 */
export function Badge({
  icon,
  label,
  colorVar,
  softVar,
  className,
  title,
}: {
  icon: string
  label: string
  colorVar: string
  softVar?: string
  className?: string
  title?: string
}) {
  return (
    <span
      title={title}
      className={cx(
        'inline-flex items-center gap-1.5 rounded-md px-2 py-1',
        'text-[11px] leading-none font-semibold whitespace-nowrap',
        className,
      )}
      style={{ color: colorVar, backgroundColor: softVar ?? 'transparent' }}
    >
      <span aria-hidden>{icon}</span>
      {label}
    </span>
  )
}

/* ------------------------------------------------------------------ states */

/**
 * Empty states are friendly and correct. "No open flags" is a *good* state
 * here, not an error, and is styled as such (§7).
 */
export function EmptyState({
  icon = '✓',
  title,
  description,
  action,
}: {
  icon?: string
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
      <div
        className="mb-4 flex h-12 w-12 items-center justify-center rounded-full text-lg"
        style={{ background: 'var(--risk-good-soft)', color: 'var(--risk-good)' }}
        aria-hidden
      >
        {icon}
      </div>
      <p className="text-[15px] font-semibold text-[var(--ink)]">{title}</p>
      <p className="mt-1.5 max-w-sm text-[13px] leading-relaxed text-[var(--ink-secondary)]">
        {description}
      </p>
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}

type BannerTone = 'warning' | 'error' | 'info'

const BANNER_TONES: Record<BannerTone, { color: string; soft: string; icon: string }> = {
  warning: { color: 'var(--risk-warning)', soft: 'var(--risk-warning-soft)', icon: '▲' },
  error: { color: 'var(--risk-critical)', soft: 'var(--risk-critical-soft)', icon: '!' },
  info: { color: 'var(--accent)', soft: 'var(--accent-soft)', icon: 'i' },
}

/** Plain language, actionable, no stack traces (§7). */
export function Banner({
  tone = 'warning',
  title,
  children,
  action,
}: {
  tone?: BannerTone
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  const style = BANNER_TONES[tone]
  return (
    <motion.div
      initial={{ opacity: 0, y: -6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -6 }}
      transition={{ duration: 0.24, ease: [0.22, 1, 0.36, 1] }}
      role={tone === 'error' ? 'alert' : 'status'}
      className="flex items-start gap-3 rounded-xl border px-4 py-3"
      style={{ borderColor: style.color, background: style.soft }}
    >
      <span
        className="mt-px flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[11px] font-bold"
        style={{ background: style.color, color: 'var(--plane)' }}
        aria-hidden
      >
        {style.icon}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-semibold" style={{ color: style.color }}>
          {title}
        </p>
        {children && (
          <div className="mt-0.5 text-[13px] leading-relaxed text-[var(--ink-secondary)]">
            {children}
          </div>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </motion.div>
  )
}

/**
 * Loading placeholders shaped like the content they replace, so the layout
 * doesn't jump when data lands - "never a blank pause" (§7).
 */
export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cx('animate-pulse rounded-md bg-[var(--surface-hover)]', className)}
      aria-hidden
    />
  )
}

export function Field({
  label,
  value,
  mono = false,
  title,
}: {
  label: string
  value: ReactNode
  mono?: boolean
  title?: string
}) {
  return (
    <div className="min-w-0">
      <p className="text-[11px] font-medium tracking-wide text-[var(--ink-muted)] uppercase">
        {label}
      </p>
      <p
        className={cx('mt-1 truncate text-[13px] text-[var(--ink)]', mono && 'identifier')}
        title={title}
      >
        {value}
      </p>
    </div>
  )
}
