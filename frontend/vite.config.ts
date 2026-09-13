import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The dashboard talks to the FastAPI service cross-origin (see the CORS
// middleware in src/api/main.py), so there is deliberately no dev proxy here:
// a proxy would hide a CORS misconfiguration in development and let it surface
// for the first time in the deployed demo.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true, // A silent port bump would land outside the CORS allowlist.
  },
  preview: {
    port: 4173,
    strictPort: true,
  },
})
