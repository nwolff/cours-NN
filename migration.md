#Assessment

Library compatibility
Library Status
@tensorflow/tfjs No Svelte dependency — fine
plotly.js-dist No Svelte dependency — fine
daisyui CSS-only — fine
tslog No Svelte dependency — fine
svelte-vega Update required: you're on v2.3.0, v4.1.0 is out with Svelte 5 support
svelte-range-slider-pips Update required: you're on v3.0.0, v4.x is out with Svelte 5 support — but see caveat below
@square/svelte-store Stores still work in Svelte 5 (the store contract is preserved), but this library appears unmaintained. You'd likely want to replace it with runes-based state eventually
What will be difficult

1. createEventDispatcher removal (4 places in NetworkGraph.svelte and DrawBox.svelte)
   The pattern dispatch('imageData', {...}) / dispatch('neuronSelected', ...) must be replaced with callback props: let { onImageData, onNeuronSelected } = $props().

2. bind:values on the range slider (your learningRates binding)
   svelte-range-slider-pips v4 works in Svelte 5, but there's a known issue with bind:values when using $state(). You'll need a workaround (an on:change handler instead of the binding).

3. Reactive statements $: — 34 instances
   These become either $derived (simple expressions) or $derived.by(() => {...}) (multi-line blocks). It's mechanical but repetitive.

4. export let → $props() — 21 instances
   All component props need to be migrated to the runes syntax: let { networkShape, activations, ... } = $props().

5. onMount/onDestroy — 18 instances
   These merge into $effect() with a cleanup return function. Not hard, but needs care with the canvas setup in DrawBox.svelte.

What's easy
{#if}, {#each}, {#await} blocks — unchanged
<slot /> → {@render children()} — one place in +layout.svelte
on:click={handler} → onclick={handler} — mostly find-and-replace
TypeScript — unchanged
CSS / Tailwind — unchanged
SvelteKit (+page.svelte, +layout.svelte, $app/paths) — unchanged
Strategy recommendation
Svelte 5 ships with a compatibility layer — you can upgrade Svelte itself without touching any component code immediately. Everything will keep working in "legacy mode". Then you migrate component by component at your own pace, starting with the simplest ones (LabelledImage, LossChart) and leaving the complex ones (DrawBox, NetworkGraph) for last.

Sources:

svelte-vega on npm
svelte-range-slider-pips GitHub
Svelte 5 migration guide
Refactoring Svelte stores to $state runes

# Phase 1

Those on:change are intentional — the RangeSlider component uses createEventDispatcher internally, so on:change is the correct way to consume it (this is the bind:values workaround from the migration plan).

The migration is complete. Here's the full summary:

All 8 files migrated:

\_Logo.svelte — export let → $props()
NetworkGraph.svelte — export let, $:, createEventDispatcher, onMount → $props(), $effect, callback prop onNeuronSelected
DrawBox.svelte — createEventDispatcher, on:event → callback prop onImageData, event handlers, kept export function clear() for bind:this
+layout.svelte, LabelledImage, LossChart, ConfusionMatrix, DataBatchGrid, DistributionChart, NetworkStats — already done in previous session
temperature_control/+page.svelte — $: → $derived/$state, $effect for slider reactivity
zero_one, all_digits, fashion, debug — same pattern, on:imageData/on:neuronSelected → callback props, on:click → onclick, bind:values → on:change workaround

# Phase 1.1

Server is up clean — no errors. App is running at http://localhost:5173/

The fixes needed were:

vega, vega-embed, vega-lite — now explicit peer deps of svelte-vega v4 (previously bundled, now separate)
svelte-range-slider-pips excluded from Vite's optimizer to avoid the Tailwind @layer base conflict

# Phase 2

The remaining diagnostics are pre-existing TypeScript issues unrelated to the migration — the validationData indexing error and onTrainEnd signature mismatch were there before. The "Cannot invoke object possibly undefined" on drawbox!.clear() is also a Svelte 5 component typing quirk.

Now done. Summary of what was fixed:

Two build warnings:

+page.svelte: <script type="ts"> → <script lang="ts">
DrawBox.svelte: let dropHighlighted = false → let dropHighlighted = $state(false)
Reactivity bugs (all 4 pages):

temperature_control: removed the $effect that caused the cycle (write→read activations in same effect), replaced with explicit oninput on sliders; fixed predict_apparent_temperature to use a local variable
All pages: replaced networkStore.update((n) => n) with \_tick++; added $derived.by(() => { \_tick; ... }) for weights and currentStats so training updates create new object references that Svelte 5 actually detects
