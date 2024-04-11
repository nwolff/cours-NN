<script lang="ts">
	import type { VegaLiteSpec } from 'svelte-vega';
	import type { EmbedOptions } from 'vega-embed';
	import { VegaLite } from 'svelte-vega';
	import { zip2 } from '$lib/generic/utils';

	export let classes: string[] = [];
	export let percentages: number[] | undefined;
	export let color = '#8888CC';
	export let highlightColor = '#0000FF';

	$: data = toData(classes, percentages, color, highlightColor);

	type DistributionData = {
		distributionData: { label: string; percentage: number; color: string }[];
	};

	const spec: VegaLiteSpec = {
		$schema: 'https://vega.github.io/schema/vega-lite/v5.json',
		width: 220,
		height: 110,
		data: {
			name: 'distributionData'
		},
		mark: 'bar',
		encoding: {
			x: {
				field: 'label',
				type: 'nominal',
				sort: null,
				title: null,
				axis: { labelFontSize: 18, tickBand: 'extent' }
			},
			y: {
				field: 'percentage',
				type: 'quantitative',
				title: null,
				scale: { domain: [0, 1] },
				axis: null
			},
			color: {
				field: 'color',
				type: 'nominal',
				scale: null
			},
			tooltip: {
				field: 'percentage',
				type: 'quantitative',
				format: '.0%',
				formatType: 'number'
			}
		}
	};

	const options: EmbedOptions = { actions: false };

	function toData(
		labels: string[],
		percentages: number[] | undefined,
		color: string,
		highlightColor: string
	): DistributionData {
		const rows = [];
		if (percentages) {
			const maxPercentage = Math.max(...percentages);
			for (const [label, percentage] of zip2(labels, percentages)) {
				rows.push({
					label: label,
					percentage: percentage,
					color: percentage == maxPercentage ? highlightColor : color
				});
			}
		} else {
			for (const label of labels) {
				rows.push({
					label: label,
					percentage: 0,
					color: 'black'
				});
			}
		}
		return { distributionData: rows };
	}
</script>

<VegaLite {data} {spec} {options} />
