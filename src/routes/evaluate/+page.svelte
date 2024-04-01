<script lang="ts">
	import { onMount } from 'svelte';
	import type { MnistData } from '$lib/data.js';
	import { mnistDataStore, networkStore } from '../../stores';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import * as tf from '@tensorflow/tfjs';
	let data: MnistData;
	let isLoading = true;
	let labelsAndPredictions: [number[], number[]];

	onMount(async () => {
		mnistDataStore.load().then((value) => {
			isLoading = false;
			data = value;
			showAccuracy();
		});
	});

	const classes = ['Zero', 'Un', 'Deux', 'Trois', 'Quatre', 'Cinq', 'Six', 'Sept', 'Huit', 'Neuf'];

	function showAccuracy() {
		const testDataSize = 1000;

		labelsAndPredictions = tf.tidy(() => {
			const testData = data.nextTestBatch(testDataSize);
			const testxs = testData.xs.reshape([testDataSize, 28 * 28]);

			const labels = testData.labels.argMax(-1);
			const preds = $networkStore.tfModel.predict(testxs).argMax(-1);

			return [labels.arraySync() as number[], preds.arraySync() as number[]];
		});
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-4 gap-4">
		<div>
			<p class="text-xl">Evaluer la précision du réseau</p>
			<br />
			<p>
				A chaque appui sur le bouton, on donne au réseau 1000 nouvelles images de <b>test</b> qu'il n'a
				jamais vues.
			</p>
			<br />
			<button class="btn btn-outline btn-primary" on:click={showAccuracy}
				>Evaluer la précision</button
			>
		</div>
		<div class="col-span-3">
			<ConfusionMatrix {classes} {labelsAndPredictions} />
		</div>
	</div>
{/if}
