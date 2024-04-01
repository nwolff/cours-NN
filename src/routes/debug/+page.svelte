<script lang="ts">
	import { onMount } from 'svelte';
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';

	import * as tf from '@tensorflow/tfjs';

	const labels = ['A', 'B', 'C', 'D'];
	const percentages = [0, 0.2, 0.5, 0.3];

	const classes = ['A', 'B', 'C'];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0],
		[0, 1, 2, 2, 2, 2]
	];

	let tfStats: string;

	onMount(async () => {
		showStats();
	});

	async function showStats() {
		tfStats = JSON.stringify(tf.memory(), null, ' ');
	}

	function handleDrawnImage(event: { detail: { image: CanvasRenderingContext2D } }) {
		console.log('handleDrawnImage', event.detail.image);
	}
</script>

<pre>
	 {tfStats}
</pre>

<div>
	<DrawBox on:imageData={handleDrawnImage} />
</div>

<div>
	<DistributionChart {labels} {percentages} color="orange" />
</div>

<div>
	<ConfusionMatrix {classes} {labelsAndPredictions} />
</div>
