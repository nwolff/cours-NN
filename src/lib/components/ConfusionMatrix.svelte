<script lang="ts">
	import type { VegaLiteSpec } from 'svelte-vega';
	import type { EmbedOptions } from 'vega-embed';
	import { VegaLite } from 'svelte-vega';

	import { zip2 } from '../utils';

	export let classes: string[] = [];
	export let labelsAndPredictions: [number[], number[]] = [[], []];

	export let width = 300;
	export let height = 300;

	$: data = toData(classes, labelsAndPredictions);

	$: options = optionsFromWidthAndHeight(width, height);

	function toData(classes: string[], [labels, predictions]: [number[], number[]]) {
		// Aggregate over each label/prediction pair
		const size = classes.length;
		const aggregates = Array(size)
			.fill(null)
			.map(() => Array(size).fill(0));
		for (const [label, prediction] of zip2(labels, predictions)) {
			aggregates[label][prediction] += 1;
		}

		// Normalize for each input label
		// (because in the data we got there were not the same number of labels shown to the network)
		// And build a datastructure that can be easily fed into vega
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

	const spec: VegaLiteSpec = {
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
				labelAngle: 0,
				labelFontSize: 17
			}
		}
	};

	function optionsFromWidthAndHeight(width: number, height: number): EmbedOptions {
		return { width: width, height: height, actions: false };
	}
</script>

<VegaLite {data} {spec} {options} />
