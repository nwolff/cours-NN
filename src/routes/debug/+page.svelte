<script lang="ts">
	import { networkStore } from '../../stores';
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { onMount } from 'svelte';
	import type { DataBatch } from '$lib/models';
	import LabelledImage from '$lib/components/LabelledImage.svelte';

	const labels = ['A', 'B', 'C', 'D'];
	const percentages = [0, 0.2, 0.5, 0.3];

	const classes = ['A', 'B', 'C'];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0],
		[0, 1, 2, 2, 2, 2]
	] as [number[], number[]];

	const batchSize = 100;
	let indexInBatch = 0;
	let dataBatch: DataBatch;

	function nextImage() {
		indexInBatch += 1;
	}

	let drawbox: DrawBox;

	function handleDrawnImage(event: { detail: { image: CanvasRenderingContext2D } }) {
		console.log('handleDrawnImage', event.detail.image);
	}

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		dataBatch = $networkStore.nextTrainBatch(batchSize);
		isLoading = false;
	});
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<LabelledImage {dataBatch} {indexInBatch} />
	<button class="btn btn-primary mt-3" on:click={nextImage}> next image </button>

	<div class="divider"></div>

	<NetworkStats stats={$networkStore.stats} />

	<div class="divider"></div>

	<ConfusionMatrix {classes} {labelsAndPredictions} width={150} height={150} />

	<div class="divider"></div>

	<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
	<button class="btn btn-primary mt-3" on:click={drawbox.clear}> clear </button>

	<div class="divider"></div>

	<DistributionChart {labels} {percentages} color="orange" />
{/if}
