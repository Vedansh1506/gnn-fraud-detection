# Dashboard

The analyst-facing frontend for the graph-based fraud detection platform. See the [root README](../README.md) for the project as a whole.

React 19 + TypeScript + Vite. It is a **thin client**: it renders what the API returns and never scores, thresholds, or decides anything itself.

## Run

```bash
npm install
npm run dev      # -> localhost:5173  (the API must be running on :8000)
```

The port matters: **5173 is in the API's CORS allowlist**, and Vite is pinned with `strictPort` so a clash fails loudly rather than drifting to 5174 and breaking only inside the browser while `curl` keeps working.

There is deliberately **no dev proxy** — a proxy would hide a CORS misconfiguration during development and surface it for the first time in the deployed demo.

```bash
npm run build    # typecheck + production bundle
npm run preview  # serve the built bundle on :4173
```

Point it at a different API with `VITE_API_BASE_URL` (compile-time — Vite has no runtime env).

## Layout

```
src/
├── lib/
│   ├── api.ts        Typed client; sorts failures into the three kinds the UI
│   │                 treats differently (expired / unreachable / degraded).
│   ├── auth.tsx      Session context. JWT in sessionStorage — tradeoff documented.
│   ├── queries.ts    TanStack Query hooks; one place decides polling cadence.
│   ├── risk.ts       THE risk-badge definition. Imported, never re-implemented.
│   ├── format.ts     Formatting. Missing values render as an em-dash.
│   └── theme.tsx     Light/dark, following the OS unless explicitly overridden.
├── components/
│   ├── ui.tsx             Card, Button, Badge, Banner, EmptyState, Skeleton.
│   ├── FlagRow.tsx        THE flag row; shares its grid template with the header.
│   ├── ShapFactors.tsx    Diverging bars — plain language first, feature name last.
│   ├── MoneyFlowGraph.tsx d3-force solves the physics, React owns the DOM.
│   ├── LiftChart.tsx      Baseline vs GNN. Zero-based axis, direct-labelled.
│   ├── StatTile.tsx       Summary tiles + animated counters.
│   └── FlowField.tsx      Login backdrop: an animated money-flow canvas.
└── routes/           Login, FlagQueue, FlagDetail, ModelOps.
```

## Conventions

These are load-bearing, not preferences:

- **Design tokens, never raw hex in components.** Colours are CSS custom properties in `index.css`, exposed to Tailwind as roles (`--ink`, `--surface`, `--risk-critical`). A hardcoded colour breaks the light/dark contract.
- **Never colour alone.** Every risk or status signal carries an icon *and* a text label — fraud UIs lean on red/green, and red/green colour blindness is common.
- **Missing data shows an em-dash.** Rendering `$0.00` or `N/A` styled as data is fabricating a value.
- **One definition per concept.** A second badge or flag-row definition is a bug, not a variant.
- **Chart colours are validated, not chosen** — the diverging poles and categorical pair were run through a colour-vision checker against the real surfaces (worst-case CVD ΔE 19.2 / 9.2). Re-run it before changing them.
- **Every data view specifies loading, empty and error states**, and one panel failing never blanks a screen.
- **Motion is subordinate to legibility**, and everything respects `prefers-reduced-motion`.

Full UI specification: `docs/Design-Document.md` (v2.0). Frontend rules: `docs/rules.md` §7.
