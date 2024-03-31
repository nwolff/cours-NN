<script lang="ts">
	import { onMount } from 'svelte';
	import { networkStore } from '../../stores';
	import * as tf from '@tensorflow/tfjs';
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import { Grid, Button, Loader, Title, Text, Divider, Stack } from '@svelteuidev/core';

	let tfvis;

	const labels = ['A', 'B', 'C', 'D'];
	const percentages = [0, 0.2, 0.5, 0.3];

	const classes = ['A', 'B', 'C'];
	const labelsAndPredictions = [
		[0, 1, 2, 1, 0, 0],
		[0, 1, 2, 2, 2, 2]
	];

	let tfStats: string;

	onMount(async () => {
		tfvis = await import('@tensorflow/tfjs-vis');
		showModelSummary();
		showStats();
	});

	async function showModelSummary() {
		const tfModel: tf.Sequential = $networkStore.tfModel;
		const summaryContainer = { name: 'Résumé du modèle', tab: 'Inspection' };
		tfvis.show.modelSummary(summaryContainer, tfModel);
		for (const [index, layer] of tfModel.layers.entries()) {
			const layerContainer = { name: 'Couche ' + index, tab: 'Inspection' };
			tfvis.show.layer(layerContainer, layer);
		}
	}

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

<!--
<DrawBox on:imageData={handleDrawnImage} />
-->
<DistributionChart {labels} {percentages} color="orange" />
<!--
<ConfusionMatrix {classes} {labelsAndPredictions} />
-->
