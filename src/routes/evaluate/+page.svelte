<script lang="ts">
	import { onMount } from 'svelte';
	import { networkStore } from '../../stores';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkStats from '$lib/components/NetworkStats.svelte';

	let labelsAndPredictions: [number[], number[]];

	$: classes = $networkStore?.shape.outputLayer.labels;
	$: testDataSize = classes?.length * 50;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		isLoading = false;
		showAccuracy();
	});

	function showAccuracy() {
		labelsAndPredictions = tf.tidy(() => {
			const testData = $networkStore.nextTestBatch(testDataSize);
			const testxs = testData.xs.reshape([testDataSize, -1]);

			const labels = testData.labels.argMax(-1);
			const preds = $networkStore.tfModel.predict(testxs).argMax(-1);

			return [labels.arraySync() as number[], preds.arraySync() as number[]];
		});
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
				A chaque appui sur le bouton, on donne au réseau {testDataSize} nouvelles images de
				<b>test</b> qu'il n'a jamais vues.
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
