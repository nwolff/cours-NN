# Remaining lint issues

ESLint is now working (`npm run lint`). The issues below were surfaced when the config was fixed and are left for a later pass.

## Dead code / unused variables

- `prediction` state variable in `all_digits`, `fashion`, and `zero_one` pages — it is assigned in the predict function but never rendered in the template. Likely leftover from a removed display widget. Safe to delete.
- `immediateNeighborsFilter` in [src/lib/LinkFilters.ts](src/lib/LinkFilters.ts) — defined but never called. Either remove it or wire it up.

## Untyped `any`

These spots use `any` explicitly, mostly around Plotly's event API and one Svelte reactive expression:

- [src/lib/components/NetworkGraph.svelte](src/lib/components/NetworkGraph.svelte) — Plotly `on('plotly_hover')` / `on('plotly_unhover')` callbacks and the `edge_x_buckets` / `edge_y_buckets` DefaultMap values
- [src/lib/components/DrawBox.svelte](src/lib/components/DrawBox.svelte) — one `any` in event handling
- [src/routes/temperature_control/+page.svelte](src/routes/temperature_control/+page.svelte) — `formatter.format(prediction as any)` in the reactive declaration

## `{#each}` blocks without keys

Svelte recommends a key expression for efficient DOM diffing:

- [src/routes/all_digits/+page.svelte](src/routes/all_digits/+page.svelte)
- [src/routes/fashion/+page.svelte](src/routes/fashion/+page.svelte)
- [src/routes/zero_one/+page.svelte](src/routes/zero_one/+page.svelte)
