<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { learningRateStore, zeroOnenetworkStore } from '../../stores';

	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { testNetwork } from '$lib/NetworkTesting';
	import DrawBox from '$lib/components/DrawBox.svelte';
	import { makeTopNLinksFilter, neighborsFilter } from '$lib/LinkFilters';
	import type { Neuron } from '$lib/NetworkShape';

	const logger = new tslog.Logger({ name: 'zero_one' });

	let learningRate = $state(0);

	const networkStore = zeroOnenetworkStore;

	const networkShape = $derived($networkStore?.shape);
	const classes = $derived($networkStore?.shape.classes);
	const weights = $derived($networkStore?.tfModel.weights);

	let drawbox = $state<DrawBox>();
	let prediction: number[] | undefined = $state(undefined);
	let activations: number[][] | undefined = $state(undefined);
	const defaultLinkFilter = makeTopNLinksFilter(700);
	let linkFilter = $state(defaultLinkFilter);
	let image: HTMLCanvasElement | undefined = $state(undefined);
	const canLearn = $derived(typeof image !== 'undefined');

	let isLoading = $state(true);
	onMount(async () => {
		await networkStore.load();
		learningRateStore.load().then((value) => {
			learningRate = value;
		});
		isLoading = false;
	});

	onDestroy(async () => {
		learningRateStore.set(learningRate);
	});

	function handleDrawnImage({ image: drawnImage }: { image: HTMLCanvasElement }) {
		image = drawnImage;
		predict_image();
	}

	function handleNeuronSelected(neuron: Neuron | null) {
		if (neuron == null) {
			linkFilter = defaultLinkFilter;
		} else {
			linkFilter = neighborsFilter(neuron);
		}
	}

	function predict_image() {
		if (image) {
			activations = calculateActivations(image);
			prediction = activations[activations.length - 1];
			logger.debug('tf.memory() ', tf.memory());
		} else {
			activations = undefined;
			prediction = undefined;
		}
	}

	function calculateActivations(image: HTMLCanvasElement): number[][] {
		return tf.tidy(() => {
			const pixels = tf.browser.fromPixels(image, 1);

			// From: https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js
			const processedImage = tf
				.reshape(pixels, [1, 28 * 28])
				.toFloat()
				.div(255);

			const activationsTensor = $networkStore.featureModel.predict(processedImage) as tf.Tensor[];
			return [processedImage, ...activationsTensor].map((x) =>
				tf.squeeze(x).arraySync()
			) as number[][];
		});
	}

	const ZERO_ACTIVATIONS = tf.tensor([[1, 0]]) as tf.Tensor2D;
	const ONE_ACTIVATIONS = tf.tensor([[0, 1]]) as tf.Tensor2D;
	const ACTIVATIONS_FOR_DIGIT = [ZERO_ACTIVATIONS, ONE_ACTIVATIONS];

	async function learn(digit: number) {
		if (!image) {
			logger.info('Cannot learn without image');
			return;
		}
		const processed_image = tf.tidy(() => {
			const pixels = tf.browser.fromPixels(image!, 1);
			return tf
				.reshape(pixels, [1, 28 * 28])
				.toFloat()
				.div(255) as tf.Tensor2D;
		});
		const trainXs = processed_image;
		const trainYs = ACTIVATIONS_FOR_DIGIT[digit];
		train(trainXs, trainYs, null, null, 1, learningRate);
	}

	async function train(
		trainXs: tf.Tensor2D,
		trainYs: tf.Tensor2D,
		valXs: tf.Tensor2D | null,
		valYs: tf.Tensor2D | null,
		batchSize: number,
		learningRate: number
	) {
		const networkUnderTraining = $networkStore;
		const optimizer = networkUnderTraining.tfModel.optimizer as tf.SGDOptimizer;
		optimizer.setLearningRate(learningRate);

		function onBatchEnd(batch: number, logs: tf.Logs) {
			logger.debug('end batch:', batch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: logs.size,
				finalAccuracy: logs.acc,
				loss: logs.loss
			});
			const testResult = testNetwork(networkUnderTraining, classes?.length * 50);
			networkUnderTraining.stats.test = testResult;
			networkStore.update((n) => n); // Notify subscribers
		}

		function onEpochEnd(epoch: number, logs: tf.Logs) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
		}

		function onTrainEnd(_logs: tf.Logs) {
			logger.debug('onTrain end : tf.memory()', tf.memory());
			tf.dispose(trainXs);
			// tf.dispose(trainYs); // For this network they live forever
			if (valXs) {
				tf.dispose(valXs);
			}
			if (valYs) {
				tf.dispose(valYs);
			}
			logger.debug('after disposing: tf.memory()', tf.memory());

			predict_image();
		}

		logger.debug('Before fit: tf.memory()', tf.memory());

		// If this fails because there is already another fit running
		// Then the 4 tensors get leaked (because the cleanup occurs in
		// onTrainEnd, which is never called)
		const params = {
			epochs: 1,
			batchSize: batchSize,
			shuffle: true,
			callbacks: { onBatchEnd, onEpochEnd, onTrainEnd }
		};
		if (valXs && valYs) {
			params['validationData'] = [valXs, valYs];
		}

		return networkUnderTraining.tfModel.fit(trainXs, trainYs, params);
	}

	async function itsAZero() {
		learn(0);
	}

	async function itsAOne() {
		learn(1);
	}

	function resetModel() {
		networkStore.reload();
		predict_image();
	}

	function clear() {
		drawbox!.clear();
		image = undefined;
		predict_image();
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-9 gap-4">
		<div class="col-span-2">
			<h4 class="text-xl mb-2">Dessiner <b>0</b> ou <b>1</b></h4>
			<DrawBox bind:this={drawbox} onImageData={handleDrawnImage} />
			<button class="btn btn-outline mt-4" disabled={!image} onclick={clear}>Effacer</button>

			<div class="divider"></div>

			<h4 class="text-xl">Apprentissage</h4>

			<ul class="menu py-4">
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" disabled={!canLearn} onclick={itsAZero}>
						C'est un 0
					</button>
				</li>
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" disabled={!canLearn} onclick={itsAOne}>
						C'est un 1
					</button>
				</li>
			</ul>

			<div class="text-l mb-2">Taux d'apprentissage</div>
			<input type="range" bind:value={learningRate} min={0} max={1} step={0.2} class="range range-primary range-xs w-full" />
			<div class="flex justify-between text-xs px-1 mt-1">
				{#each [0, 0.2, 0.4, 0.6, 0.8, 1] as v}
					<span class:font-bold={learningRate === v}>{v}</span>
				{/each}
			</div>
		</div>
		<div class="col-span-5">
			<NetworkGraph
				{networkShape}
				{weights}
				{activations}
				{linkFilter}
				onNeuronSelected={handleNeuronSelected}
			/>
		</div>
		<div class="col-span-2">
			<NetworkStats stats={$networkStore.stats} />
			<div class="m-6"></div>
			<button class="btn btn-outline btn-error" onclick={resetModel}>
				Réinitialiser le réseau
			</button>
		</div>
	</div>
{/if}
