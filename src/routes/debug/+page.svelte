<script lang="ts">
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import { temperatureControlNetworkStore } from '../../stores';
	import { onMount } from 'svelte';

	const classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
	const percentages = [0, 0.2, 0.5, 0.3, 0.1, 0.3];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0, 3, 4, 5, 6, 7],
		[0, 1, 2, 2, 2, 2, 4, 4, 4, 6, 6]
	] as [number[], number[]];

	let drawbox: DrawBox;
	const previewCanvasSize = 28 * 4;

	const networkStore = temperatureControlNetworkStore;

	//
	$: dataBatch = $networkStore?.nextTrainBatch(100);
	let indexInBatch = 0;
	$: x = dataBatch?.xs.slice([indexInBatch], [1]);
	$: y = dataBatch?.ys.slice([indexInBatch], [1]);

	let processedImage: HTMLCanvasElement;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		isLoading = false;
	});

	function handleDrawnImage(event: { detail: { image: HTMLCanvasElement } }) {
		const ctx = processedImage.getContext('2d')!;
		ctx.drawImage(event.detail.image, 0, 0, previewCanvasSize, previewCanvasSize);
	}

	function clear() {
		drawbox.clear();
		const ctx = processedImage.getContext('2d')!;
		ctx.clearRect(0, 0, previewCanvasSize, previewCanvasSize);
	}

	function nextInBatch() {
		indexInBatch += 1;
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="flex">
		<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
		<canvas
			class="border-2 m-4"
			width={previewCanvasSize}
			height={previewCanvasSize}
			bind:this={processedImage}
		></canvas>
	</div>
	<button class="btn btn-primary mt-3" on:click={clear}>Clear</button>

	<div class="divider"></div>

	Index in batch {indexInBatch}
	<dl>
		<dt><b>x</b></dt>
		<dd>
			shape: {x.shape}
			<br />
			{x}
		</dd>
		<dt><b>y</b></dt>
		<dd>
			shape: {y.shape}
			<br />
			{y}
		</dd>
	</dl>
	<button class="btn btn-primary mt-3" on:click={nextInBatch}>Next</button>

	<div class="divider"></div>

	<ConfusionMatrix {classes} {labelsAndPredictions} />

	<ConfusionMatrix {classes} {labelsAndPredictions} size={140} />

	<div class="divider"></div>

	<DistributionChart {classes} {percentages} color="orange" />
{/if}
