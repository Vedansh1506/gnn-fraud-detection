/**
 * Login (Design Doc §4.1).
 *
 * Centred card, username + password, one Sign in button. No sign-up, no forgot
 * password - single-persona MVP. Failure shows an inline, plain-language error
 * under the form; it never says whether the username or the password was the
 * wrong one, matching the API, which returns the same 401 for both.
 */

import { useState, type FormEvent } from 'react'
import { motion } from 'motion/react'
import { useAuth } from '../lib/auth'
import { ApiError, API_BASE_URL, messageFor } from '../lib/api'
import { Button } from '../components/ui'
import { FlowField } from '../components/FlowField'

export function Login() {
  const { signIn } = useAuth()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)

  async function onSubmit(event: FormEvent) {
    event.preventDefault()
    if (!username.trim() || !password) {
      setError('Enter a username and password.')
      return
    }

    setSubmitting(true)
    setError(null)
    try {
      await signIn(username.trim(), password)
    } catch (caught) {
      setError(
        caught instanceof ApiError && caught.status === 401
          ? 'Invalid username or password.'
          : messageFor(caught),
      )
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="relative flex min-h-full items-center justify-center overflow-hidden bg-[var(--plane)] px-4">
      <FlowField />
      {/* Keeps the form legible over the animation without hiding it. */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 52% 44% at 50% 50%, var(--plane) 0%, transparent 100%)',
        }}
        aria-hidden
      />

      <motion.div
        initial={{ opacity: 0, y: 14, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="relative w-full max-w-[380px]"
      >
        <div className="mb-7 text-center">
          <svg
            width="34"
            height="34"
            viewBox="0 0 24 24"
            fill="none"
            className="mx-auto mb-4"
            aria-hidden
          >
            <path d="M6 6 L18 12 L6 18" stroke="var(--accent)" strokeWidth="1.5" opacity="0.45" />
            <circle cx="6" cy="6" r="2.6" fill="var(--accent)" />
            <circle cx="18" cy="12" r="2.6" fill="var(--risk-critical)" />
            <circle cx="6" cy="18" r="2.6" fill="var(--accent)" />
          </svg>
          <h1 className="text-[19px] font-semibold tracking-tight text-[var(--ink)]">
            Mule Detection Platform
          </h1>
          <p className="mt-1.5 text-[13px] text-[var(--ink-secondary)]">
            Graph-based fraud review for analysts
          </p>
        </div>

        <form
          onSubmit={onSubmit}
          className="rounded-[var(--radius-card)] border border-[var(--border)] bg-[var(--surface)] p-6"
          style={{ boxShadow: 'var(--shadow-card)' }}
        >
          <LabelledInput
            id="username"
            label="Username"
            value={username}
            onChange={setUsername}
            autoComplete="username"
            autoFocus
          />
          <div className="h-4" />
          <LabelledInput
            id="password"
            label="Password"
            type="password"
            value={password}
            onChange={setPassword}
            autoComplete="current-password"
          />

          <Button
            type="submit"
            variant="primary"
            loading={submitting}
            className="mt-6 w-full py-2.5"
          >
            {submitting ? 'Signing in…' : 'Sign in'}
          </Button>

          {error && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mt-3 text-center text-[13px]"
              style={{ color: 'var(--risk-critical)' }}
              role="alert"
            >
              {error}
            </motion.p>
          )}
        </form>

        <p className="identifier mt-5 text-center text-[11px] text-[var(--ink-muted)]">
          {API_BASE_URL}
        </p>
      </motion.div>
    </div>
  )
}

function LabelledInput({
  id,
  label,
  value,
  onChange,
  type = 'text',
  autoComplete,
  autoFocus,
}: {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  type?: string
  autoComplete?: string
  autoFocus?: boolean
}) {
  return (
    <div>
      <label
        htmlFor={id}
        className="mb-1.5 block text-[12px] font-medium text-[var(--ink-secondary)]"
      >
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        autoComplete={autoComplete}
        autoFocus={autoFocus}
        onChange={(event) => onChange(event.target.value)}
        className="w-full rounded-lg border border-[var(--border-strong)] bg-[var(--plane)] px-3 py-2.5 text-[14px] text-[var(--ink)] transition-colors duration-[var(--duration-fast)] outline-none placeholder:text-[var(--ink-muted)] focus:border-[var(--accent)]"
      />
    </div>
  )
}
