<script lang="ts">
	import { onMount } from 'svelte';
	import { networkStore } from '../../stores';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { zip2 } from '$lib/generic/utils';

	let labelsAndPredictions: [number[], number[]];

	$: classes = $networkStore?.shape.outputLayer.labels;
	$: testDataSize = classes?.length * 50;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		isLoading = false;
		showConfusionMatrixAndAccuracy();
	});

	//	type TestResult { }

	function showConfusionMatrixAndAccuracy() {
		labelsAndPredictions = tf.tidy(() => {
			const testData = $networkStore.nextTestBatch(testDataSize);
			const testxs = testData.xs.reshape([testDataSize, -1]);

			const labels = testData.labels.argMax(-1);
			const preds = $networkStore.tfModel.predict(testxs).argMax(-1);

			return [labels.arraySync() as number[], preds.arraySync() as number[]];
		});
		$networkStore.stats.testAccuracy = computeAccuracy(...labelsAndPredictions);
	}

	function computeAccuracy(labels: number[], predictions: number[]): number {
		let allPredictions = 0;
		let correctPredictions = 0;
		for (const [label, prediction] of zip2(labels, predictions)) {
			allPredictions += 1;
			if (label == prediction) {
				correctPredictions += 1;
			}
		}
		return correctPredictions / allPredictions;
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
			<button class="btn btn-outline btn-primary" on:click={showConfusionMatrixAndAccuracy}>
				Evaluer la précision
			</button>
			<NetworkStats stats={$networkStore.stats} />
		</div>
		<div class="col-span-5">
			<ConfusionMatrix {classes} {labelsAndPredictions} />
			<ConfusionMatrix size={150} {classes} {labelsAndPredictions} />
		</div>
	</div>
{/if}
