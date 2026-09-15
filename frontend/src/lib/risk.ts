/**
 * One definition of risk presentation, imported everywhere (Design Doc §4.5).
 *
 * The queue, the detail header and the graph all read from this file, so a
 * badge cannot mean one thing on one screen and something else on another.
 *
 * **What these bands are and are not.** They are *display* bands for directing
 * attention within an already-flagged queue - they are not calibrated risk
 * tiers, and no evaluation produced them. The only threshold in this system
 * with evidence behind it is the flag threshold itself (0.9, an operational
 * choice about queue volume, served by /models). Everything above that
 * threshold is, by the model's own account, worth a human look; these bands
 * just order that work. Labelled "Very high"/"High" rather than
 * "90% likely fraud" for exactly that reason - the score is not a probability
 * of fraud and must never be presented as one.
 */

export type RiskLevel = 'critical' | 'high' | 'below'

export interface RiskBand {
  level: RiskLevel
  label: string
  /** Never colour alone (§11): every badge carries an icon and a word. */
  icon: string
  colorVar: string
  softVar: string
}

const CRITICAL_AT = 0.97

export function riskBand(score: number, flagThreshold = 0.9): RiskBand {
  if (score >= CRITICAL_AT) {
    return {
      level: 'critical',
      label: 'Very high',
      icon: '●',
      colorVar: 'var(--risk-critical)',
      softVar: 'var(--risk-critical-soft)',
    }
  }
  if (score >= flagThreshold) {
    return {
      level: 'high',
      label: 'High',
      icon: '▲',
      colorVar: 'var(--risk-warning)',
      softVar: 'var(--risk-warning-soft)',
    }
  }
  return {
    level: 'below',
    label: 'Below threshold',
    icon: '○',
    colorVar: 'var(--ink-muted)',
    softVar: 'transparent',
  }
}

export interface DecisionBadge {
  label: string
  icon: string
  colorVar: string
  softVar: string
}

export function decisionBadge(decision: string | null): DecisionBadge | null {
  if (decision === 'confirmed_fraud') {
    return {
      label: 'Confirmed fraud',
      icon: '✓',
      colorVar: 'var(--risk-critical)',
      softVar: 'var(--risk-critical-soft)',
    }
  }
  if (decision === 'false_positive') {
    return {
      label: 'Dismissed',
      icon: '✓',
      colorVar: 'var(--risk-good)',
      softVar: 'var(--risk-good-soft)',
    }
  }
  return null
}

/**
 * Which way a SHAP factor pushed.
 *
 * Positive contributions raise the score, negative lower it. Saying "raises" /
 * "lowers" rather than showing a signed number is the plain-language rule
 * (Principle 2) - the number is still shown alongside for anyone who wants it.
 */
export function contributionDirection(contribution: number): {
  label: string
  colorVar: string
} {
  return contribution >= 0
    ? { label: 'raises risk', colorVar: 'var(--viz-pos)' }
    : { label: 'lowers risk', colorVar: 'var(--viz-neg)' }
}
