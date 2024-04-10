<script lang="ts">
	import DrawBox from '$lib/components/DrawBox.svelte';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { Link } from '$lib/NetworkShape';
	import * as tf from '@tensorflow/tfjs';
	import { networkStore } from '../../stores';
	import { onMount } from 'svelte';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';

	const logger = new tslog.Logger({ name: 'trainmanually' });

	$: networkShape = $networkStore?.shape;
	$: weights = $networkStore?.tfModel.weights;

	let drawbox;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		isLoading = false;
	});

	function handleDrawnImage(event: { detail: { image: ImageData } }) {
		const image = event.detail.image;
		const processed_image = tf.tidy(() => {
			const pixels = tf.browser.fromPixels(image, 1);

			// XXX: This leaks
			return tf
				.reshape(pixels, [1, 28 * 28])
				.toFloat()
				.div(255);
		});
		logger.debug('tf.memory() ', tf.memory());
		train(processed_image);
	}

	async function train(processed_image: tf.Tensor) {
		const trainXs = processed_image;
		const trainYs = tf.tensor([[1, 0]]);
		const networkUnderTraining = $networkStore;
		networkUnderTraining.tfModel.optimizer = new tf.SGDOptimizer(1); // XXX: constant

		function onEpochEnd(epoch: number, logs?: tf.Logs) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: 1,
				finalAccuracy: logs?.val_acc
			});
			networkStore.update((n) => n); // Notify subscribers
		}

		function onTrainEnd(logs?: tf.Logs) {
			logger.debug('on train end: tf.memory()', tf.memory());
			tf.dispose(trainXs);
			tf.dispose(trainYs);
			logger.debug('after disposing: tf.memory()', tf.memory());
		}

		return networkUnderTraining.tfModel.fit(trainXs, trainYs, {
			validationData: [trainXs, trainYs],
			epochs: 1,
			batchSize: 1,
			callbacks: { onTrainEnd, onEpochEnd }
		});
	}

	function resetModel() {
		networkStore.reload();
	}

	function linkFilter(links: Link[]) {
		const length = links.length;
		if (length <= 500) {
			return links;
		}
		const sortedLinks = [...links].sort(
			(l1: Link, l2: Link) => Math.abs(l2.weight) - Math.abs(l1.weight)
		);
		return sortedLinks.slice(0, Math.min(500, 0.1 * length));
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-7 gap-4">
		<div class="col-span-2">
			<h4 class="text-xl mb-2">Dessiner un chiffre</h4>
			<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
			<button class="btn btn-outline btn-primary mt-4" on:click={drawbox.clear}>Effacer</button>
			<ul class="menu py-4">
				<li class="mt-8">
					<button class="btn btn-outline btn-error" on:click={resetModel}>
						Réinitialiser le réseau
					</button>
				</li>
			</ul>
			<NetworkStats stats={$networkStore.stats} />
		</div>
		<div class="col-span-5">
			<NetworkGraph {networkShape} {weights} {linkFilter} />
		</div>
	</div>
{/if}
