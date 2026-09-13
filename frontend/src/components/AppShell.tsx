/**
 * Sidebar shell (Design Doc §3).
 *
 * Two top-level destinations only - Flag Queue and Model & Ops. Flag Detail is
 * a drill-down from the queue, never a nav item, because a third entry would
 * ask the analyst to choose where to go instead of showing them their work.
 */

import { NavLink, useLocation } from 'react-router-dom'
import { motion } from 'motion/react'
import { useAuth } from '../lib/auth'
import { useTheme } from '../lib/theme'
import { useHealth } from '../lib/queries'
import { Badge, Button, cx } from './ui'

const NAV = [
  { to: '/queue', label: 'Flag Queue', icon: '◧', operatorOnly: false },
  { to: '/ops', label: 'Model & Ops', icon: '◈', operatorOnly: false },
]

export function AppShell({ children }: { children: React.ReactNode }) {
  const { session, signOut, isOperator } = useAuth()
  const location = useLocation()

  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)] md:flex">
        <Wordmark />

        <nav className="flex flex-col gap-1 px-3 py-2">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} className="relative">
              {({ isActive }) => (
                <span
                  className={cx(
                    'relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium',
                    'transition-colors duration-[var(--duration-fast)]',
                    isActive
                      ? 'text-[var(--ink)]'
                      : 'text-[var(--ink-secondary)] hover:bg-[var(--surface-hover)] hover:text-[var(--ink)]',
                  )}
                >
                  {isActive && (
                    // One shared element slides between items rather than each
                    // one fading independently - it reads as a single cursor
                    // moving, which is the point of the animation.
                    <motion.span
                      layoutId="nav-active"
                      className="absolute inset-0 rounded-lg bg-[var(--accent-soft)]"
                      transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                    />
                  )}
                  <span className="relative text-[var(--accent)]" aria-hidden>
                    {item.icon}
                  </span>
                  <span className="relative">{item.label}</span>
                </span>
              )}
            </NavLink>
          ))}
        </nav>

        <div className="mt-auto flex flex-col gap-3 border-t border-[var(--border)] p-3">
          <ServiceStatus />
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">
              <p className="truncate text-[13px] font-medium text-[var(--ink)]">
                {session?.username}
              </p>
              <p className="text-[11px] text-[var(--ink-muted)] capitalize">
                {isOperator ? 'Operator' : 'Analyst'}
              </p>
            </div>
            <ThemeToggle />
          </div>
          <Button variant="ghost" onClick={signOut} className="justify-start">
            Sign out
          </Button>
        </div>
      </aside>

      {/* Narrow viewports get the nav as a top bar. Desktop-first by design
          (§8) - this is a graceful fallback, not a phone layout. */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between gap-3 border-b border-[var(--border)] bg-[var(--surface)] px-4 py-2 md:hidden">
          <div className="flex gap-1">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  cx(
                    'rounded-lg px-3 py-1.5 text-[13px] font-medium',
                    isActive
                      ? 'bg-[var(--accent-soft)] text-[var(--ink)]'
                      : 'text-[var(--ink-secondary)]',
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Button variant="ghost" onClick={signOut}>
              Sign out
            </Button>
          </div>
        </div>

        <main key={location.pathname} className="min-w-0 flex-1 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  )
}

function Wordmark() {
  return (
    <div className="flex items-center gap-2.5 px-5 py-5">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden>
        {/* A three-node money-flow motif: the product's actual subject, rather
            than a generic shield icon. */}
        <path d="M6 6 L18 12 L6 18" stroke="var(--accent)" strokeWidth="1.5" opacity="0.45" />
        <circle cx="6" cy="6" r="2.6" fill="var(--accent)" />
        <circle cx="18" cy="12" r="2.6" fill="var(--risk-critical)" />
        <circle cx="6" cy="18" r="2.6" fill="var(--accent)" />
      </svg>
      <div className="leading-tight">
        <p className="text-[13px] font-semibold text-[var(--ink)]">Mule Detection</p>
        <p className="text-[11px] text-[var(--ink-muted)]">Graph + GNN scoring</p>
      </div>
    </div>
  )
}

/**
 * Honest states (Principle 4): this reports what the service says about itself,
 * including "can't reach it at all", rather than assuming green.
 */
function ServiceStatus() {
  const { data, isError, isLoading } = useHealth()

  if (isLoading) {
    return <p className="px-1 text-[11px] text-[var(--ink-muted)]">Checking service…</p>
  }
  if (isError || !data) {
    return (
      <Badge
        icon="!"
        label="Service unreachable"
        colorVar="var(--risk-critical)"
        softVar="var(--risk-critical-soft)"
      />
    )
  }

  const tone =
    data.status === 'ok'
      ? { icon: '●', color: 'var(--risk-good)', soft: 'var(--risk-good-soft)', label: 'All systems ok' }
      : data.status === 'degraded'
        ? {
            icon: '▲',
            color: 'var(--risk-warning)',
            soft: 'var(--risk-warning-soft)',
            label: 'Degraded',
          }
        : {
            icon: '!',
            color: 'var(--risk-critical)',
            soft: 'var(--risk-critical-soft)',
            label: 'Unhealthy',
          }

  return (
    <div className="flex flex-col gap-1.5">
      <Badge icon={tone.icon} label={tone.label} colorVar={tone.color} softVar={tone.soft} />
      <p className="identifier px-1 text-[11px] text-[var(--ink-muted)]">{data.model_version}</p>
    </div>
  )
}

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <button
      onClick={toggle}
      aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
      className="flex h-8 w-8 items-center justify-center rounded-lg text-[var(--ink-secondary)] transition-colors hover:bg-[var(--surface-hover)] hover:text-[var(--ink)]"
    >
      <span aria-hidden>{theme === 'dark' ? '☾' : '☀'}</span>
    </button>
  )
}
