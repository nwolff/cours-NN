<script lang="ts">
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import DataBatchGrid from '$lib/components/DataBatchGrid.svelte';
	import LossChart from '$lib/components/LossChart.svelte';
	import { onMount } from 'svelte';
	import { allDigitsNetworkStore } from '../../stores';

	const classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
	const percentages = [0, 0.2, 0.5, 0.3, 0.1, 0.3];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0, 3, 4, 5, 6, 7],
		[0, 1, 2, 2, 2, 2, 4, 4, 4, 6, 6]
	] as [number[], number[]];

	let drawbox = $state<DrawBox>();
	const previewCanvasSize = 28 * 4;

	let processedImage: HTMLCanvasElement;

	const networkStore = allDigitsNetworkStore;

	import type { DataBatch } from '$lib/DataSource';
	let dataBatch: DataBatch | null = $state(null);

	onMount(async () => {
		await networkStore.load();
		dataBatch = $networkStore.nextTestBatch(10);
		// isLoading = false;
	});

	function handleDrawnImage({ image }: { image: HTMLCanvasElement }) {
		const ctx = processedImage.getContext('2d')!;
		ctx.drawImage(image, 0, 0, previewCanvasSize, previewCanvasSize);
	}

	function clear() {
		drawbox!.clear();
		const ctx = processedImage.getContext('2d')!;
		ctx.clearRect(0, 0, previewCanvasSize, previewCanvasSize);
	}

	const losses = [
		{ samples: 1, loss: 5 },
		{ samples: 2, loss: 15 },
		{ samples: 3, loss: 8 },
		{ samples: 4, loss: 20 }
	];
</script>

<div class="flex">
	<DrawBox bind:this={drawbox} onImageData={handleDrawnImage} />
	<canvas
		class="border-2 m-4"
		width={previewCanvasSize}
		height={previewCanvasSize}
		bind:this={processedImage}
	></canvas>
</div>
<button class="btn btn-primary mt-3" onclick={clear}>Clear</button>

<div class="divider"></div>

<LossChart {losses} />

<div class="divider"></div>

<ConfusionMatrix {classes} {labelsAndPredictions} />

<ConfusionMatrix {classes} {labelsAndPredictions} size={140} />

<div class="divider"></div>

<DistributionChart {classes} {percentages} color="orange" />

<div class="divider"></div>

{#if dataBatch}<DataBatchGrid {dataBatch}></DataBatchGrid>{/if}
