<script lang="ts">
	import { onMount } from 'svelte';
	import type { MnistData } from '$lib/data.js';
	import { mnistDataStore, networkStore } from '../../stores';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import * as tf from '@tensorflow/tfjs';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';

	const logger = new tslog.Logger({ name: 'evaluate' });

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

		logger.debug('tf.memory() ', tf.memory());
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-7 gap-4">
		<div class="col-span-2">
			<p class="text-xl">Evaluer la précision du réseau</p>
			<br />
			<p>
				A chaque appui sur le bouton, on donne au réseau 1000 nouvelles images de <b>test</b> qu'il n'a
				jamais vues.
			</p>
			<br />
			<button class="btn btn-outline btn-primary" on:click={showAccuracy}>
				Evaluer la précision
			</button>
			<NetworkStats stats={$networkStore.stats} />
		</div>
		<div class="col-span-5">
			<ConfusionMatrix {classes} {labelsAndPredictions} />
		</div>
	</div>
{/if}
