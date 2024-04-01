<script lang="ts">
	import type { MnistData } from '$lib/data.js';
	import type { Link } from '$lib/NetworkShape';
	import { onDestroy, onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { learningRateStore, mnistDataStore, networkStore } from '../../stores';
	import RangeSlider from 'svelte-range-slider-pips';
	import { newAllDigitsNetwork } from '$lib/models';

	const networkShape = $networkStore.shape;
	let data: MnistData;
	let isLoading = true;
	let learningRates = [0];

	let tfvis;

	let fitCallbacksContainer: HTMLElement;

	$: weights = $networkStore.tfModel.weights;

	onMount(async () => {
		tfvis = await import('@tensorflow/tfjs-vis');

		mnistDataStore.load().then((value) => {
			isLoading = false;
			data = value;
		});

		learningRateStore.load().then((value) => {
			learningRates = [value];
		});
	});

	onDestroy(async () => {
		learningRateStore.set(learningRates[0]);
	});

	async function train({
		trainDataSize = 5000,
		batchSize = 100,
		epochs = 1,
		learningRate = 0.01
	} = {}) {
		const networkUnderTraining = $networkStore;

		networkUnderTraining.tfModel.optimizer = new tf.SGDOptimizer(learningRate);

		const testDataSize = trainDataSize / 5;

		const [trainXs, trainYs] = tf.tidy(() => {
			const d = data.nextTrainBatch(trainDataSize);
			return [d.xs.reshape([trainDataSize, 28 * 28]), d.labels];
		});

		const [testXs, testYs] = tf.tidy(() => {
			const d = data.nextTestBatch(testDataSize);
			return [d.xs.reshape([testDataSize, 28 * 28]), d.labels];
		});

		const metrics = ['acc', 'val_acc'];
		const visualFitCallbacks = tfvis.show.fitCallbacks(fitCallbacksContainer, metrics);

		function onBatchEnd(batch: number, logs: any): Promise<void> {
			networkUnderTraining.trainingRoundDone({ samplesSeen: 100, finalAccuracy: 1 }); // XXX
			networkStore.update((n) => n); // Just to notify the views
			return visualFitCallbacks.onBatchEnd(batch, logs);
		}

		return networkUnderTraining.tfModel.fit(trainXs, trainYs, {
			validationData: [testXs, testYs],
			epochs: epochs,
			batchSize: batchSize,
			shuffle: true,
			callbacks: { onBatchEnd }
		});
	}

	async function train100() {
		train({ trainDataSize: 100, batchSize: 25, epochs: 1, learningRate: learningRates[0] });
	}

	async function train1000() {
		train({ trainDataSize: 1000, batchSize: 50, epochs: 1, learningRate: learningRates[0] });
	}

	async function train5000() {
		train({ trainDataSize: 5000, batchSize: 100, epochs: 1, learningRate: learningRates[0] });
	}

	/*
	async function trainFully() {
		train({ trainDataSize: 5000, batchSize: 100, epochs: 8, learningRate: learningRates[0] });
	}
	*/

	function resetModel() {
		networkStore.update(() => newAllDigitsNetwork()); // XXX
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
	<div class="grid grid-cols-4 gap-4">
		<div>
			<h4 class="text-xl">Entraîner avec des exemples</h4>
			<br />
			<p>Taux d'apprentissage</p>
			<RangeSlider
				bind:values={learningRates}
				min={0}
				max={1}
				step={0.2}
				pips
				all="label"
				springValues={{ stiffness: 0.2, damping: 0.7 }}
			/>

			<div>
				<ul class="menu pt-4">
					<li class="mt-1">
						<button class="btn btn-outline btn-primary" on:click={train100}>
							Entraîner avec 100 images
						</button>
					</li>
					<li class="mt-1">
						<button class="btn btn-outline btn-primary" on:click={train1000}>
							Entraîner avec 1'000 images
						</button>
					</li>
					<li class="mt-1">
						<button class="btn btn-outline btn-primary" on:click={train5000}>
							Entraîner avec 5'000 images
						</button>
					</li>
					<li class="mt-8">
						<button class="btn btn-outline btn-error" on:click={resetModel}>
							Réinitialiser le réseau
						</button>
					</li>
				</ul>
			</div>

			<div bind:this={fitCallbacksContainer} />
		</div>
		<div class="col-span-3">
			<NetworkGraph {networkShape} {weights} {linkFilter} />
		</div>
	</div>
{/if}
