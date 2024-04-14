<script lang="ts">
	import type { VegaLiteSpec } from 'svelte-vega';
	import { VegaLite } from 'svelte-vega';

	import { zip2 } from '$lib/generic/utils';

	export let classes: string[] = [];
	export let labelsAndPredictions: [number[], number[]] = [[], []];

	export let size = 300;

	$: data = toData(classes, labelsAndPredictions);
	$: spec = size <= 150 ? miniSpec : fullSpec;
	$: options = { width: size, height: size, actions: false };

	// Note that some labels may have been shown more often than others to the network
	function toData(classes: string[], [labels, predictions]: [number[], number[]]) {
		// Count occurences of every label/prediction pair
		const size = classes.length;
		const aggregates = Array(size)
			.fill(null)
			.map(() => Array(size).fill(0));
		for (const [label, prediction] of zip2(labels, predictions)) {
			aggregates[label][prediction] += 1;
		}

		// MatrixData contains everything needed by vega.
		// We build a signedPercentage field that is positive only when on the diagonal
		const matrixData = [];
		for (const [colIndex, col] of aggregates.entries()) {
			const colSum = col.reduce((a, b) => a + b, 0) as number;
			for (const [rowIndex, element] of col.entries()) {
				const percentage = element / colSum;
				matrixData.push({
					actual: classes[colIndex],
					predicted: classes[rowIndex],
					percentage: percentage,
					signedPercentage: rowIndex == colIndex ? percentage : -percentage
				});
			}
		}
		return { matrixData: matrixData };
	}

	const fullSpec: VegaLiteSpec = {
		$schema: 'https://vega.github.io/schema/vega-lite/v5.json',
		data: { name: 'matrixData' },
		width: 400,
		height: 400,
		encoding: {
			x: {
				field: 'actual',
				type: 'nominal',
				sort: null,
				title: 'Classe réelle',
				axis: {
					orient: 'top'
				}
			},
			y: {
				field: 'predicted',
				type: 'nominal',
				sort: null,
				title: 'Prédiction'
			}
		},
		layer: [
			{
				mark: 'rect',
				params: [
					{
						name: 'select-col',
						select: {
							type: 'point',
							fields: ['actual'],
							on: 'pointerover'
						}
					}
				],
				encoding: {
					color: {
						field: 'signedPercentage',
						type: 'quantitative',
						scale: {
							range: ['red', 'white', 'blue'],
							interpolate: 'hsl',
							domain: [-1, 0, 1]
						},
						legend: null
					}
				}
			},
			{
				mark: {
					type: 'text',
					fontSize: 13
				},
				encoding: {
					text: {
						field: 'percentage',
						type: 'quantitative',
						format: '.0%',
						formatType: 'number'
					},
					color: {
						condition: { test: "datum['percentage'] < 0.4", value: 'black' },
						value: 'white'
					},
					opacity: {
						condition: { param: 'select-col', empty: false, value: 1 },
						value: 0
					}
				}
			}
		],
		config: {
			axis: {
				titlePadding: 15,
				titleFontSize: 20,
				tickBand: 'extent',
				labelFontSize: 17
			}
		}
	};

	const miniSpec: VegaLiteSpec = {
		$schema: 'https://vega.github.io/schema/vega-lite/v5.json',
		data: { name: 'matrixData' },
		width: 400,
		height: 400,
		background: null,
		mark: 'rect',
		encoding: {
			x: {
				field: 'actual',
				type: 'nominal',
				sort: null,
				title: null
			},
			y: {
				field: 'predicted',
				type: 'nominal',
				sort: null,
				title: null
			},
			color: {
				field: 'signedPercentage',
				type: 'quantitative',
				scale: {
					range: ['red', 'white', 'blue'],
					interpolate: 'hsl',
					domain: [-1, 0, 1]
				},
				legend: null
			}
		},
		config: {
			axis: {
				labelFontSize: 14,
				tickBand: 'extent'
			}
		}
	};
</script>

<VegaLite {data} {spec} {options} />
