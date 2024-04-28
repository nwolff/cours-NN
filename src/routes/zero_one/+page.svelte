<script lang="ts">
	import { Link } from '$lib/NetworkShape';
	import { onDestroy, onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { learningRateStore, zeroOnenetworkStore } from '../../stores';
	import RangeSlider from 'svelte-range-slider-pips';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { testNetwork } from '$lib/NetworkTesting';
	import DrawBox from '$lib/components/DrawBox.svelte';

	const logger = new tslog.Logger({ name: 'zero_one' });

	let learningRates = [0];

	const networkStore = zeroOnenetworkStore;

	$: networkShape = $networkStore?.shape;
	$: classes = $networkStore?.shape.classes;
	$: weights = $networkStore?.tfModel.weights;

	let drawbox: DrawBox;
	let prediction: number[] | undefined;
	let activations: number[][] | undefined;
	let linkFilter = keepTopLinks;
	let image: ImageData | undefined;

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
		image = event.detail.image;
		predict_image();
	}

	function predict_image() {
		if (image) {
			activations = calculateActivations(image);
			prediction = activations[activations.length - 1];
			linkFilter = activatedlinkFilter;
			logger.debug('tf.memory() ', tf.memory());
		} else {
			linkFilter = keepTopLinks;
			activations = undefined;
			prediction = undefined;
		}
	}

	function calculateActivations(image: ImageData): number[][] {
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

	function applyActivation(links: Link[]): Link[] {
		if (!links.find((link) => link.a.activation)) {
			// A small optimization
			return links;
		}
		return links.map(
			(link) => new Link(link.a, link.b, link.weight * (1 + 0.5 * link.a.activation))
		);
	}

	function activatedlinkFilter(links: Link[]) {
		const linksWithActivationApplied = applyActivation(links);
		return keepTopLinks(linksWithActivationApplied);
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

			// XXX: This leaks ?
			return tf
				.reshape(pixels, [1, 28 * 28])
				.toFloat()
				.div(255) as tf.Tensor2D;
		});
		const trainXs = processed_image;
		const trainYs = ACTIVATIONS_FOR_DIGIT[digit];
		train(trainXs, trainYs, null, null, 1, learningRates[0]);
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
			if (logs?.val_acc) {
				networkUnderTraining.trainingRoundDone({
					samplesSeen: 0,
					finalAccuracy: logs.val_acc
				});
				networkStore.update((n) => n); // Notify subscribers
			}
		}

		function onTrainEnd(_logs?: tf.Logs) {
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
		drawbox.clear();
		image = undefined;
		predict_image();
	}

	function keepTopLinks(links: Link[]) {
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
			<h4 class="text-xl mb-2">Dessiner <b>0</b> ou <b>1</b></h4>
			<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
			<button class="btn btn-outline btn-primary mt-4" on:click={clear}>Effacer</button>

			<div class="divider"></div>

			<h4 class="text-xl">Apprentissage</h4>

			<ul class="menu py-4">
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={itsAZero}> C'est un 0 </button>
				</li>
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={itsAOne}> C'est un 1 </button>
				</li>
			</ul>

			<div class="text-l mb-2">Taux d'apprentissage</div>
			<RangeSlider
				bind:values={learningRates}
				min={0}
				max={1}
				step={0.2}
				pips
				all="label"
				springValues={{ stiffness: 0.2, damping: 0.7 }}
			/>
		</div>
		<div class="col-span-5">
			<NetworkGraph {networkShape} {weights} {activations} {linkFilter} />
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
