<script lang="ts">
	import { onMount } from 'svelte';
	import { modelStore } from '../../stores';
	import DistributionChart from '$lib/DistributionChart.svelte';
	import DrawBox from '$lib/DrawBox.svelte';
	import ConfusionMatrix from '$lib/ConfusionMatrix.svelte';

	let tfvis;

	const labels = ['One', 'Two', 'Three', 'Four'];
	const values = [0, 0.1, 0.2, 0.3];

	const classes = ['A', 'B', 'C'];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0],
		[0, 1, 2, 2, 2, 2]
	];

	onMount(async () => {
		tfvis = await import('@tensorflow/tfjs-vis');
		showModelSummary();
	});

	async function showModelSummary() {
		const summaryContainer = { name: 'Résumé du modèle', tab: 'Inspection' };
		tfvis.show.modelSummary(summaryContainer, $modelStore);
		for (const [index, layer] of $modelStore.layers.entries()) {
			const layerContainer = { name: 'Couche ' + index, tab: 'Inspection' };
			tfvis.show.layer(layerContainer, layer);
		}
	}

	function handleDrawnImage(event: { detail: { image: CanvasRenderingContext2D } }) {
		console.log('handleDrawnImage', event.detail.image);
	}
</script>

<DrawBox on:imageData={handleDrawnImage} />
<DistributionChart {labels} {values} color="orange" />
<ConfusionMatrix {classes} {labelsAndPredictions} />
