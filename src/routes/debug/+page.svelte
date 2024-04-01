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
	] as [number[], number[]];

	let drawbox: DrawBox;

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

<div>
	<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
	<button class="btn btn-primary mt-3" on:click={drawbox.clear}> clear </button>
</div>

<div class="divider"></div>

<div>
	<ConfusionMatrix {classes} {labelsAndPredictions} width={150} height={150} />
</div>

<div class="divider"></div>

<div>
	<DistributionChart {labels} {percentages} color="orange" />
</div>

<div class="divider"></div>

<pre>
	 {tfStats}
</pre>
