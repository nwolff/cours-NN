<script lang="ts">
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';

	const classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];
	const percentages = [0, 0.2, 0.5, 0.3, 0.1, 0.3];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0, 3, 4, 5, 6, 7],
		[0, 1, 2, 2, 2, 2, 4, 4, 4, 6, 6]
	] as [number[], number[]];

	let drawbox: DrawBox;
	const previewCanvasSize = 28 * 4;

	let processedImage: HTMLCanvasElement;

	function handleDrawnImage(event: { detail: { image: HTMLCanvasElement } }) {
		const ctx = processedImage.getContext('2d')!;
		ctx.drawImage(event.detail.image, 0, 0, previewCanvasSize, previewCanvasSize);
	}

	function clear() {
		drawbox.clear();
		const ctx = processedImage.getContext('2d')!;
		ctx.clearRect(0, 0, previewCanvasSize, previewCanvasSize);
	}
</script>

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

<ConfusionMatrix {classes} {labelsAndPredictions} />

<ConfusionMatrix {classes} {labelsAndPredictions} size={140} />

<div class="divider"></div>

<DistributionChart {classes} {percentages} color="orange" />
