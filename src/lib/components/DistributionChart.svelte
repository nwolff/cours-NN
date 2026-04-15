<script lang="ts">
	import type { VegaLiteSpec } from 'svelte-vega';
	import type { EmbedOptions } from 'vega-embed';
	import { VegaLite } from 'svelte-vega';
	import { zip2 } from '$lib/generic/utils';

	let {
		classes = [],
		rotateClassNames = false,
		percentages,
		color = '#8888CC',
		highlightColor = '#0000FF'
	}: {
		classes?: string[];
		rotateClassNames?: boolean;
		percentages?: number[];
		color?: string;
		highlightColor?: string;
	} = $props();

	const data = $derived(toData(classes, percentages, color, highlightColor));
	const spec = $derived(makeSpec(rotateClassNames));

	type DistributionData = {
		distributionData: { label: string; percentage: number; color: string }[];
	};

	type WithXLabelAngle = { encoding: { x: { axis: { labelAngle: number } } } };

	function makeSpec(rotateClassNames: boolean) {
		const spec = structuredClone(defaultSpec);
		if (rotateClassNames) {
			(spec as WithXLabelAngle).encoding.x.axis.labelAngle = 90;
		}
		return spec;
	}

	const defaultSpec: VegaLiteSpec = {
		$schema: 'https://vega.github.io/schema/vega-lite/v6.json',
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
				axis: { labelFontSize: 18, labelAngle: 0, tickBand: 'extent' }
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
				rows.push({ label: label, percentage: 0, color: 'black' });
			}
		}
		return { distributionData: rows };
	}
</script>

<VegaLite {data} {spec} {options} />
