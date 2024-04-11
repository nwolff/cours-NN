<script lang="ts">
	import type { Link } from '$lib/NetworkShape';
	import { onDestroy, onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { learningRateStore, networkStore } from '../../stores';
	import RangeSlider from 'svelte-range-slider-pips';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { testNetwork } from '$lib/NetworkTesting';
	import DrawBox from '$lib/components/DrawBox.svelte';

	const logger = new tslog.Logger({ name: 'train' });

	let learningRates = [0];

	$: networkShape = $networkStore?.shape;
	$: classes = $networkStore?.shape.classes;
	$: weights = $networkStore?.tfModel.weights;

	let drawbox: DrawBox;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		learningRateStore.load().then((value) => {
			learningRates = [value];
		});
		isLoading = false;
	});

	onDestroy(async () => {
		learningRateStore.set(learningRates[0]);
	});

	function handleDrawnImage(event: { detail: { image: ImageData } }) {
		const image = event.detail.image;
		const processed_image = tf.tidy(() => {
			const pixels = tf.browser.fromPixels(image, 1);

			// XXX: This leaks ?
			return tf
				.reshape(pixels, [1, 28 * 28])
				.toFloat()
				.div(255) as tf.Tensor2D;
		});
		logger.debug('tf.memory() ', tf.memory());
		const trainXs = processed_image;
		const trainYs = tf.tensor([[1, 0]]) as tf.Tensor2D;
		train(trainXs, trainYs, trainXs, trainYs, 1, learningRates[0]);
	}

	async function trainOnData(trainDataSize: number, batchSize: number) {
		logger.debug('before generating train data: tf.memory()', tf.memory());
		const [trainXs, trainYs] = tf.tidy(() => {
			const d = $networkStore.nextTrainBatch(trainDataSize);
			return [d.xs.reshape([trainDataSize, 28 * 28]), d.labels];
		});
		const validationDataSize = Math.ceil(trainDataSize / 5);
		const [valXs, valYs] = tf.tidy(() => {
			const d = $networkStore.nextTrainBatch(validationDataSize);
			return [d.xs.reshape([validationDataSize, 28 * 28]), d.labels];
		});
		logger.debug('after generating train data: tf.memory()', tf.memory());
		train(trainXs, trainYs, valXs, valYs, batchSize, learningRates[0]);
	}

	async function train(
		trainXs: tf.Tensor2D,
		trainYs: tf.Tensor2D,
		valXs: tf.Tensor2D,
		valYs: tf.Tensor2D,
		batchSize: number,
		learningRate: number
	) {
		const networkUnderTraining = $networkStore;
		networkUnderTraining.tfModel.optimizer = new tf.SGDOptimizer(learningRate);

		function onBatchEnd(batch: number, logs?: tf.Logs) {
			logger.debug('end batch:', batch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: logs?.size || 0,
				finalAccuracy: logs?.acc
			});
			const testResult = testNetwork(networkUnderTraining, classes?.length * 50);
			networkUnderTraining.stats.test = testResult;
			networkStore.update((n) => n); // Notify subscribers
		}

		function onEpochEnd(epoch: number, logs?: tf.Logs) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: 0,
				finalAccuracy: logs?.val_acc
			});
			networkStore.update((n) => n); // Notify subscribers
		}

		function onTrainEnd(logs?: tf.Logs) {
			logger.debug('onTrain end : tf.memory()', tf.memory());
			tf.dispose(trainXs);
			tf.dispose(trainYs);
			tf.dispose(valXs);
			tf.dispose(valYs);
			logger.debug('after disposing: tf.memory()', tf.memory());
		}

		logger.debug('Before fit: tf.memory()', tf.memory());

		// If this fails because there is already another fit running
		// Then the 4 tensors get leaked (because the cleanup occurs in
		// onTrainEnd, which is never called)
		return networkUnderTraining.tfModel.fit(trainXs, trainYs, {
			validationData: [valXs, valYs],
			epochs: 1,
			batchSize: batchSize,
			shuffle: true,
			callbacks: { onBatchEnd, onTrainEnd, onEpochEnd }
		});
	}

	async function train1() {
		trainOnData(1, 1);
	}

	async function train10() {
		trainOnData(10, 10);
	}

	async function train100() {
		trainOnData(100, 50);
	}

	async function train1000() {
		trainOnData(1000, 50);
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
	<div class="grid grid-cols-9 gap-4">
		<div class="col-span-2">
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
			<h4 class="text-xl mb-2">Dessiner un chiffre</h4>
			<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
			<button class="btn btn-outline btn-primary mt-4" on:click={drawbox.clear}>Effacer</button>
			<h4 class="text-xl">Entraîner avec des exemples</h4>
			<br />
			<ul class="menu py-4">
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train1}>
						Entraîner avec 1 image
					</button>
				</li>
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train10}>
						Entraîner avec 10 images
					</button>
				</li>
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
			</ul>
		</div>
		<div class="col-span-5">
			<NetworkGraph {networkShape} {weights} {linkFilter} />
		</div>
		<div class="col-span-2">
			<NetworkStats stats={$networkStore.stats} />
			<ul class="menu py-10 mx-5">
				<li>
					<button class="btn btn-outline btn-error" on:click={resetModel}>
						Réinitialiser le réseau
					</button>
				</li>
			</ul>
		</div>
	</div>
{/if}
