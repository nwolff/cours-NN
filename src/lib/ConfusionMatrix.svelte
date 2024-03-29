<script lang="ts">
	import type { VegaLiteSpec } from 'svelte-vega';
	import type { EmbedOptions } from 'vega-embed';
	import { VegaLite } from 'svelte-vega';

	export let classes: string[] = [];
	export let labelsAndPredictions: [number[], number[]] = [[], []];

	$: data = toData(classes, labelsAndPredictions);

	function toData(classes: string[], [labels, predictions]: [number[], number[]]) {
		// Aggregate over each label/prediction pair
		const size = classes.length;
		const aggregates = Array(size)
			.fill(null)
			.map(() => Array(size).fill(0));
		for (const [i, label] of labels.entries()) {
			const prediction = predictions[i];
			aggregates[label][prediction] += 1;
		}

		// Normalize for each input label
		// (because in the data we got there were not the same number of labels shown to the network)
		// And build a datastructure that can be easily fed into vega
		const matrixData = [];
		for (const [rowIndex, row] of aggregates.entries()) {
			const rowSum = row.reduce((a, b) => a + b, 0) as number;
			for (const [colIndex, element] of row.entries()) {
				matrixData.push({
					actual: classes[rowIndex],
					predicted: classes[colIndex],
					percentage: element / rowSum
				});
			}
		}
		return { matrixData: matrixData };
	}

	const spec: VegaLiteSpec = {
		$schema: 'https://vega.github.io/schema/vega-lite/v5.json',
		width: 400,
		height: 400,
		data: {
			name: 'matrixData'
		},
		mark: 'rect',
		encoding: {
			x: {
				field: 'predicted',
				type: 'nominal',
				sort: null,
				title: 'Chiffre prédit'
			},
			y: {
				field: 'actual',
				type: 'nominal',
				sort: null,
				title: 'Chiffre montré'
			},
			fill: {
				field: 'percentage',
				title: "Nombre d'occurences",
				type: 'quantitative',
				scale: {
					range: ['white', 'blue'],
					interpolate: 'hsl',
					domain: [0, 1]
				},
				legend: null
			},
			tooltip: {
				field: 'percentage',
				type: 'quantitative',
				format: '.0%',
				formatType: 'number'
			}
		},
		config: {
			axis: {
				titlePadding: 15,
				titleFontSize: 18,
				grid: true,
				tickBand: 'extent',
				labelAngle: 0,
				labelFontSize: 12
			}
		}
	};

	const options: EmbedOptions = { actions: false };
</script>

<VegaLite {data} {spec} {options} />
