/**
 * Session handling for the dashboard.
 *
 * **Where the token lives, and the honest tradeoff.** The JWT is kept in
 * `sessionStorage`, so it survives a page refresh but dies when the tab closes.
 * The genuinely more secure option is an httpOnly cookie the browser will not
 * hand to JavaScript, which would make this immune to token theft via XSS -
 * that needs the API to set and read cookies plus CSRF protection, which is a
 * backend change beyond the MVP. `sessionStorage` over `localStorage` is the
 * meaningful part of the choice: it bounds exposure to the life of the tab
 * rather than leaving a bearer token on disk indefinitely. Tokens also expire
 * in 60 minutes server-side. Documented as a known MVP limitation rather than
 * presented as best practice.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { api, ApiError } from './api'

const STORAGE_KEY = 'gnn-fraud.session'

export interface Session {
  token: string
  username: string
  role: string
  /** Epoch ms. Used only to drop an obviously-dead token before sending it. */
  expiresAt: number
}

interface AuthValue {
  session: Session | null
  isOperator: boolean
  signIn: (username: string, password: string) => Promise<void>
  signOut: () => void
  /** Called when any request comes back 401, so one expired token clears the app. */
  handleUnauthorized: (error: unknown) => void
}

const AuthContext = createContext<AuthValue | null>(null)

function readStoredSession(): Session | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Session
    // A token past its expiry is worse than none: it produces a confusing 401
    // on first load instead of a clean login screen.
    if (!parsed.token || parsed.expiresAt < Date.now()) return null
    return parsed
  } catch {
    // Private mode, cleared storage, or a corrupt value - all mean "no session".
    return null
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(readStoredSession)

  useEffect(() => {
    try {
      if (session) sessionStorage.setItem(STORAGE_KEY, JSON.stringify(session))
      else sessionStorage.removeItem(STORAGE_KEY)
    } catch {
      // Storage being unavailable degrades to an in-memory session, which is
      // a worse experience but a working one.
    }
  }, [session])

  const signIn = useCallback(async (username: string, password: string) => {
    const response = await api.login(username, password)
    setSession({
      token: response.access_token,
      username,
      role: response.role,
      expiresAt: Date.now() + response.expires_in_minutes * 60_000,
    })
  }, [])

  const signOut = useCallback(() => setSession(null), [])

  const handleUnauthorized = useCallback((error: unknown) => {
    if (error instanceof ApiError && error.kind === 'unauthorized') setSession(null)
  }, [])

  const value = useMemo<AuthValue>(
    () => ({
      session,
      isOperator: session?.role === 'operator',
      signIn,
      signOut,
      handleUnauthorized,
    }),
    [session, signIn, signOut, handleUnauthorized],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside AuthProvider')
  return value
}

/** The token, for callers that are only rendered behind a signed-in route. */
export function useToken(): string {
  const { session } = useAuth()
  return session?.token ?? ''
}
