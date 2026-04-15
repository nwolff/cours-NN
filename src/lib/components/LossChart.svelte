<script lang="ts">
	import type { Losses } from '$lib/Network';
	import type { VegaLiteSpec } from 'svelte-vega';
	import { VegaLite } from 'svelte-vega';

	let {
		losses,
		width = 200,
		height = 100
	}: { losses: Losses; width?: number; height?: number } = $props();

	const dataObject = $derived(toDataObject(losses));
	const options = $derived({ width, height, actions: false });

	function toDataObject(losses: Losses) {
		return { table: losses };
	}

	const spec: VegaLiteSpec = {
		$schema: 'https://vega.github.io/schema/vega-lite/v6.json',
		data: { name: 'table' },

		mark: {
			type: 'area'
		},
		encoding: {
			x: { field: 'samples', type: 'quantitative', axis: { title: null } },
			y: {
				field: 'loss',
				type: 'quantitative',
				scale: { domainMin: 0 },
				axis: { title: null }
			},
			tooltip: [
				{ field: 'samples', type: 'quantitative' },
				{ field: 'loss', type: 'quantitative' }
			],
			opacity: { value: 0.6 }
		},
		config: {
			view: { stroke: null } // Removes the outer border box
		}
	};
</script>

<VegaLite data={dataObject} {spec} {options} />
