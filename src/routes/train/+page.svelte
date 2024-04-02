<script lang="ts">
	import type { MnistData } from '$lib/data.js';
	import type { Link } from '$lib/NetworkShape';
	import { onDestroy, onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { learningRateStore, mnistDataStore, networkStore } from '../../stores';
	import RangeSlider from 'svelte-range-slider-pips';
	import { newAllDigitsNetwork } from '$lib/models';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';

	const logger = new tslog.Logger({ name: 'train' });

	const networkShape = $networkStore.shape;
	let data: MnistData;
	let isLoading = true;
	let learningRates = [0];

	$: weights = $networkStore.tfModel.weights;

	onMount(async () => {
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

		const validationDataSize = trainDataSize / 5;

		logger.debug('before generating train data: tf.memory()', tf.memory());
		const [trainXs, trainYs] = tf.tidy(() => {
			const d = data.nextTrainBatch(trainDataSize);
			return [d.xs.reshape([trainDataSize, 28 * 28]), d.labels];
		});
		logger.debug('after generating train data: tf.memory()', tf.memory());

		logger.debug('before generating test data: tf.memory()', tf.memory());
		const [validationXs, validationYs] = tf.tidy(() => {
			const d = data.nextTestBatch(validationDataSize);
			return [d.xs.reshape([validationDataSize, 28 * 28]), d.labels];
		});
		logger.debug('after generating test data: tf.memory()', tf.memory());

		function onBatchEnd(batch: number, logs?: tf.Logs) {
			logger.debug('end batch:', batch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: logs?.size || 0,
				finalAccuracy: logs?.acc
			});
			networkStore.update((n) => n); // Notify subscribers
		}

		function onEpochEnd(epoch: number, logs?: tf.Logs) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: 0,
				finalAccuracy: logs?.val_acc // XXX: Should acc and val_acc be kept separate in networkUnderTraining ?
			});
			networkStore.update((n) => n); // Notify subscribers
		}

		function onTrainEnd(logs?: tf.Logs) {
			logger.debug('on train end. logs:', logs);
			logger.debug('onTrain end : tf.memory()', tf.memory());
			tf.dispose(trainXs);
			tf.dispose(trainYs);
			tf.dispose(validationXs);
			tf.dispose(validationYs);
			logger.debug('after disposing: tf.memory()', tf.memory());
		}

		logger.debug('Before fit: tf.memory()', tf.memory());

		// If this fails because there is already another fit running
		// Then the 4 tensors get leaked (because the cleanup occurs in
		// onTrainEnd, which is never called)
		return networkUnderTraining.tfModel.fit(trainXs, trainYs, {
			validationData: [validationXs, validationYs],
			epochs: epochs,
			batchSize: batchSize,
			shuffle: true,
			callbacks: { onBatchEnd, onTrainEnd, onEpochEnd }
		});
	}

	async function train100() {
		train({ trainDataSize: 100, batchSize: 50, epochs: 1, learningRate: learningRates[0] });
	}

	async function train1000() {
		train({ trainDataSize: 1000, batchSize: 50, epochs: 1, learningRate: learningRates[0] });
	}

	async function train5000() {
		train({ trainDataSize: 5000, batchSize: 50, epochs: 1, learningRate: learningRates[0] });
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
	<div class="grid grid-cols-7 gap-4">
		<div class="col-span-2">
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

			<ul class="menu py-4">
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
			<NetworkStats stats={$networkStore.stats} />
		</div>
		<div class="col-span-5">
			<NetworkGraph {networkShape} {weights} {linkFilter} />
		</div>
	</div>
{/if}
