/**
 * Routing and the signed-in/signed-out split.
 *
 * There is no route guard component: the app renders the login screen whenever
 * there is no session, so there is no protected route that can be reached with
 * a missing token. A 401 from any request clears the session (see auth.tsx),
 * which lands the analyst back here rather than on a screen full of errors.
 */

import { Navigate, Route, Routes } from 'react-router-dom'
import { useAuth } from './lib/auth'
import { AppShell } from './components/AppShell'
import { Login } from './routes/Login'
import { FlagQueue } from './routes/FlagQueue'
import { ModelOps } from './routes/ModelOps'

export default function App() {
  const { session } = useAuth()

  if (!session) return <Login />

  return (
    <AppShell>
      <Routes>
        <Route path="/queue" element={<FlagQueue />} />
        <Route path="/ops" element={<ModelOps />} />
        <Route path="*" element={<Navigate to="/queue" replace />} />
      </Routes>
    </AppShell>
  )
}
