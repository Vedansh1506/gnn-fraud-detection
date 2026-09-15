/**
 * The missing-data contract.
 *
 * The rule these enforce (`rules.md` §7.2, Design Doc §7): a value the system
 * does not have renders as an em-dash. Never a zero, never "N/A" styled to look
 * like data. Audit rows written before the transaction-detail columns existed
 * genuinely have no amount, and showing "$0.00" there would be inventing one —
 * in a fraud tool, where the amount is what an analyst triages on.
 */

import { describe, expect, it } from 'vitest'
import {
  MISSING,
  formatAmount,
  formatCount,
  formatDateTime,
  formatMetric,
  formatPercent,
  formatScore,
  shortAccount,
} from './format'

describe('the missing-data contract', () => {
  it('renders a missing amount as an em-dash, not a zero', () => {
    expect(formatAmount(null, 'US Dollar')).toBe(MISSING)
  })

  it('does not treat a genuine zero as missing', () => {
    // A real zero-value transaction is data, not absence. Collapsing the two
    // would hide a real (and odd) transaction behind the same glyph.
    expect(formatAmount(0, 'US Dollar')).not.toBe(MISSING)
    expect(formatAmount(0, 'US Dollar')).toContain('0')
  })

  it('renders missing timestamps, accounts and metrics as em-dashes', () => {
    expect(formatDateTime(null)).toBe(MISSING)
    expect(shortAccount(null)).toBe(MISSING)
    expect(formatMetric(null)).toBe(MISSING)
    expect(formatPercent(null)).toBe(MISSING)
  })

  it('renders an unparseable timestamp as an em-dash rather than "Invalid Date"', () => {
    expect(formatDateTime('not-a-date')).toBe(MISSING)
  })
})

describe('amounts', () => {
  it('formats a known currency with its symbol', () => {
    expect(formatAmount(4791.42, 'US Dollar')).toContain('4,791.42')
  })

  it('keeps the currency visible when the name is unrecognised', () => {
    // The dataset spans a dozen currencies, so a bare number is ambiguous.
    // Dropping the unit would be worse than an unfamiliar label.
    const formatted = formatAmount(1000, 'Some Unknown Currency')
    expect(formatted).toContain('1,000')
    expect(formatted).toContain('Some Unknown Currency')
  })
})

describe('account keys', () => {
  it('leaves short keys intact', () => {
    expect(shortAccount('020_80011D730')).toBe('020_80011D730')
  })

  it('truncates long keys but keeps both ends recognisable', () => {
    const long = '0224_800127B30AAAAAAAAAAAAAAAA'
    const short = shortAccount(long)

    expect(short.length).toBeLessThan(long.length)
    // Both ends survive: the identifier is the thing being investigated, so a
    // truncation that hides which account this is would defeat the point.
    expect(short.startsWith('0224_80')).toBe(true)
    expect(short.endsWith(long.slice(-6))).toBe(true)
  })
})

describe('numeric display', () => {
  it('shows scores at three decimals so near-threshold flags are distinguishable', () => {
    // 0.90 vs 0.904 matters when the flag threshold is 0.9.
    expect(formatScore(0.9042)).toBe('0.904')
  })

  it('shows AUPRC at enough precision to be meaningful', () => {
    // These values are ~0.02; two decimals would render the headline as "0.02"
    // for both models and erase the entire comparison.
    expect(formatMetric(0.021791703643701366)).toBe('0.0218')
    expect(formatMetric(0.019756724619549915)).toBe('0.0198')
    expect(formatMetric(0.0218)).not.toBe(formatMetric(0.0198))
  })

  it('formats counts with thousands separators', () => {
    expect(formatCount(1174673)).toBe('1,174,673')
  })

  it('formats a rate as a percentage', () => {
    expect(formatPercent(0.158)).toBe('15.8%')
  })
})
