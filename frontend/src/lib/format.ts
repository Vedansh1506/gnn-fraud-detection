/**
 * Display formatting.
 *
 * The rule running through this file: when a value is missing, say so with an
 * em-dash. Never substitute a zero, a placeholder amount or "N/A" styled to
 * look like data - audit rows written before the queue columns existed
 * genuinely have no amount, and showing "$0.00" would be inventing one.
 */

export const MISSING = '—'

/** IBM AML currency names ("US Dollar") rather than ISO codes, so a lookup. */
const CURRENCY_CODES: Record<string, string> = {
  'US Dollar': 'USD',
  Euro: 'EUR',
  'UK Pound': 'GBP',
  Yen: 'JPY',
  'Swiss Franc': 'CHF',
  'Australian Dollar': 'AUD',
  'Canadian Dollar': 'CAD',
  'Indian Rupee': 'INR',
  Yuan: 'CNY',
  'Mexican Peso': 'MXN',
  'Brazil Real': 'BRL',
  'Saudi Riyal': 'SAR',
  Shekel: 'ILS',
  'Ruble': 'RUB',
  'Rupee': 'INR',
}

export function formatAmount(amount: number | null, currency: string | null): string {
  if (amount === null || amount === undefined) return MISSING

  const code = currency ? CURRENCY_CODES[currency] : undefined
  if (code) {
    try {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: code,
        maximumFractionDigits: 2,
      }).format(amount)
    } catch {
      // An unknown code must not break the row it appears in.
    }
  }
  const formatted = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(amount)
  // Name the currency rather than dropping it - "12,345.67" alone is ambiguous
  // in a dataset that spans a dozen currencies.
  return currency ? `${formatted} ${currency}` : formatted
}

/** Compact form for tight cells and axis labels: 12.3K, 1.2M. */
export function formatCompact(value: number): string {
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(
    value,
  )
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat('en-US').format(value)
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return MISSING
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return MISSING
  return new Intl.DateTimeFormat('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date)
}

/**
 * "3 minutes ago". Used for `scored_at` (when we saw it), never for the
 * transaction time - the dataset is from 2022, so "4 years ago" on every row
 * would be noise rather than information.
 */
export function formatRelative(iso: string | null): string {
  if (!iso) return MISSING
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return MISSING

  const seconds = Math.round((date.getTime() - Date.now()) / 1000)
  const formatter = new Intl.RelativeTimeFormat('en', { numeric: 'auto' })
  const divisions: [number, Intl.RelativeTimeFormatUnit][] = [
    [60, 'second'],
    [60, 'minute'],
    [24, 'hour'],
    [7, 'day'],
    [4.35, 'week'],
    [12, 'month'],
  ]

  let value = seconds
  for (const [step, unit] of divisions) {
    if (Math.abs(value) < step) return formatter.format(Math.round(value), unit)
    value /= step
  }
  return formatter.format(Math.round(value), 'year')
}

/**
 * `account_key` is bank+account. Shortening keeps the queue scannable, but the
 * full value is always available in a title attribute so nothing is lost -
 * truncation that hides an identifier with no way to recover it is a real
 * problem when the identifier is what you are investigating.
 */
export function shortAccount(accountKey: string | null): string {
  if (!accountKey) return MISSING
  if (accountKey.length <= 16) return accountKey
  return `${accountKey.slice(0, 7)}…${accountKey.slice(-6)}`
}

export function formatScore(score: number): string {
  return score.toFixed(3)
}

export function formatPercent(value: number | null, digits = 1): string {
  if (value === null || value === undefined) return MISSING
  return `${(value * 100).toFixed(digits)}%`
}

/** AUPRC values here are ~0.02, so the usual 2 decimal places would show 0.02. */
export function formatMetric(value: number | null, digits = 4): string {
  if (value === null || value === undefined) return MISSING
  return value.toFixed(digits)
}
