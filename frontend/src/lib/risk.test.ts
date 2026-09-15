/**
 * The risk-presentation contract.
 *
 * Two rules are load-bearing and easy to break silently in a refactor:
 *
 * 1. **Never colour alone.** Every risk state carries an icon *and* a word.
 *    Fraud UIs lean on red/green, and red/green colour blindness is common —
 *    a badge that degrades to "a coloured dot" is unreadable for those users.
 * 2. **The bands are display ordering, not calibrated probabilities.** The
 *    labels must never read as a likelihood of fraud.
 */

import { describe, expect, it } from 'vitest'
import { contributionDirection, decisionBadge, riskBand } from './risk'

const FLAG_THRESHOLD = 0.9

describe('never colour alone', () => {
  it('gives every risk band an icon and a word, not just a colour', () => {
    for (const score of [0.99, 0.95, 0.5]) {
      const band = riskBand(score, FLAG_THRESHOLD)

      expect(band.icon).toBeTruthy()
      expect(band.label).toBeTruthy()
      expect(band.colorVar).toBeTruthy()
    }
  })

  it('gives every decision badge an icon and a word', () => {
    for (const decision of ['confirmed_fraud', 'false_positive']) {
      const badge = decisionBadge(decision)

      expect(badge).not.toBeNull()
      expect(badge!.icon).toBeTruthy()
      expect(badge!.label).toBeTruthy()
    }
  })

  it('distinguishes bands by icon, not only by colour', () => {
    // If two bands shared an icon, a colourblind user would see them as
    // identical — which is exactly the failure this rule exists to prevent.
    const icons = [riskBand(0.99), riskBand(0.95), riskBand(0.5)].map((b) => b.icon)
    expect(new Set(icons).size).toBe(3)
  })
})

describe('risk bands', () => {
  it('separates very high from high', () => {
    expect(riskBand(0.99, FLAG_THRESHOLD).level).toBe('critical')
    expect(riskBand(0.95, FLAG_THRESHOLD).level).toBe('high')
  })

  it('marks anything under the flag threshold as below it', () => {
    expect(riskBand(0.89, FLAG_THRESHOLD).level).toBe('below')
  })

  it('treats a score exactly at the threshold as flagged', () => {
    // The API flags on `score >= threshold`; the UI must not disagree with it
    // at the boundary, or a flagged row renders as "below threshold".
    expect(riskBand(0.9, FLAG_THRESHOLD).level).not.toBe('below')
  })

  it('follows the threshold it is given rather than a hardcoded one', () => {
    // The threshold is operator-configurable and served by /models.
    expect(riskBand(0.75, 0.7).level).not.toBe('below')
    expect(riskBand(0.75, 0.8).level).toBe('below')
  })

  it('never labels a band as a probability of fraud', () => {
    // These are display bands for ordering attention. No evaluation produced
    // them, so a label like "90% likely fraud" would assert something nothing
    // has earned — and the score is not a fraud probability.
    for (const score of [0.99, 0.95, 0.5]) {
      const label = riskBand(score, FLAG_THRESHOLD).label.toLowerCase()
      expect(label).not.toMatch(/%|percent|probab|likel|certain/)
    }
  })
})

describe('decisions', () => {
  it('returns nothing for an unreviewed flag, which is what makes it open', () => {
    expect(decisionBadge(null)).toBeNull()
  })

  it('does not invent a badge for an unrecognised decision', () => {
    expect(decisionBadge('something_else')).toBeNull()
  })

  it('distinguishes confirmed fraud from a dismissal', () => {
    expect(decisionBadge('confirmed_fraud')!.label).not.toBe(
      decisionBadge('false_positive')!.label,
    )
  })
})

describe('SHAP contribution direction', () => {
  it('describes direction in plain words rather than a bare sign', () => {
    expect(contributionDirection(1.2).label).toMatch(/raises/i)
    expect(contributionDirection(-0.4).label).toMatch(/lowers/i)
  })

  it('uses opposite colours for the two directions', () => {
    // The diverging chart depends on this: same colour both ways would make
    // the sign — the most useful part for an analyst — invisible.
    expect(contributionDirection(1).colorVar).not.toBe(contributionDirection(-1).colorVar)
  })

  it('treats zero as non-negative rather than crashing', () => {
    expect(contributionDirection(0).label).toBeTruthy()
  })
})
