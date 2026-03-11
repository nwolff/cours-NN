<script lang="ts">
	import type { Losses } from '$lib/Network';
	import type { VegaLiteSpec } from 'svelte-vega';
	import { VegaLite } from 'svelte-vega';

	export let losses: Losses;
	export let width = 200;
	export let height = 100;

	$: dataObject = toDataObject(losses);

	function toDataObject(losses: Losses) {
		return { table: losses };
	}

	$: options = { width: width, height: height, actions: false };

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
