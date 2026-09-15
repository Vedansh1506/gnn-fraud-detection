/**
 * Conditional className joiner.
 *
 * Lives in its own module rather than alongside the components in `ui.tsx`:
 * a file that exports both components and plain functions breaks React Fast
 * Refresh, which oxlint flags (`react/only-export-components`).
 */
export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(' ')
}
