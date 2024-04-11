<script lang="ts">
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import type { DataBatch } from '$lib/DataSource';
	import LabelledImage from '$lib/components/LabelledImage.svelte';
	import * as tslog from 'tslog';
	import * as tf from '@tensorflow/tfjs';

	const logger = new tslog.Logger({ name: 'debug' });

	const percentages = [0, 0.2, 0.5, 0.3, 0.1, 0.3];

	const classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0, 3, 4, 5, 6, 7],
		[0, 1, 2, 2, 2, 2, 4, 4, 4, 6, 6]
	] as [number[], number[]];

	let drawbox: DrawBox;

	function handleDrawnImage(event: { detail: { image: CanvasRenderingContext2D } }) {
		console.log('handleDrawnImage', event.detail.image);
	}
</script>

<ConfusionMatrix {classes} {labelsAndPredictions} />

<div class="divider"></div>

<ConfusionMatrix {classes} {labelsAndPredictions} size={140} />

<div class="divider"></div>

<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
<button class="btn btn-primary mt-3" on:click={drawbox.clear}> clear </button>

<div class="divider"></div>

<DistributionChart {classes} {percentages} color="orange" />
