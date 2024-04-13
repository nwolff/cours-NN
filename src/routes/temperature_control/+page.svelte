<script lang="ts">
	import { Link } from '$lib/NetworkShape';
	import { onDestroy, onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { learningRateStore, temperatureControlNetworkStore } from '../../stores';
	import RangeSlider from 'svelte-range-slider-pips';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { testNetwork } from '$lib/NetworkTesting';
	import {
		MIN_TEMP,
		MAX_TEMP,
		tempToActivation,
		humidityToActivation,
		MIN_HUMIDITY,
		MAX_HUMIDITY,
		activationToHeatIndex,
		computeHeatIndex
	} from '$lib/networks/TemperatureControlNetwork';

	const logger = new tslog.Logger({ name: 'temperature' });

	let learningRates = [0];

	const networkStore = temperatureControlNetworkStore;

	$: networkShape = $networkStore?.shape;
	$: classes = $networkStore?.shape.classes;
	$: weights = $networkStore?.tfModel.weights;

	$: temperature = MIN_TEMP;
	$: relativeHumidity = MIN_HUMIDITY;

	let prediction: number[] | undefined;
	let activations: number[][] | undefined;
	let linkFilter = keepTopLinks;

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	$: formattedPrediction = formatter.format(activationToHeatIndex(prediction));

	$: formattedComputedHeatIndex = formatter.format(computeHeatIndex(temperature, relativeHumidity));

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

	function setTemperature(event: Event) {
		temperature = event.target?.valueAsNumber;
		predict_control();
	}

	function setRelativeHumidity(event: Event) {
		relativeHumidity = event.target?.valueAsNumber;
		predict_control();
	}

	function predict_control() {
		activations = calculateActivations(temperature, relativeHumidity);
		prediction = activations[activations.length - 1];
		linkFilter = activatedlinkFilter;
		logger.debug('tf.memory() ', tf.memory());
	}

	function calculateActivations(temperature: number, relativeHumidity: number): number[][] {
		return tf.tidy(() => {
			let input = tf
				.tensor([tempToActivation(temperature), humidityToActivation(relativeHumidity)])
				.reshape([-1, 2]);
			const activationTensor = $networkStore.featureModel.predict(input) as tf.Tensor[];
			return [input, activationTensor].map((x) => tf.squeeze(x).arraySync()) as number[][];
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

	async function trainOnData(trainDataSize: number, batchSize: number) {
		const trainData = $networkStore.nextTrainBatch(trainDataSize);
		const validationDataSize = Math.ceil(trainDataSize / 20);
		const valData = $networkStore.nextTrainBatch(validationDataSize);
		train(trainData.xs, trainData.ys, valData.xs, valData.ys, batchSize, learningRates[0]);
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

		function onBatchBegin(batch: number, logs?: tf.Logs) {
			predict_control();
		}

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

		function onTrainEnd(logs?: tf.Logs) {
			logger.debug('onTrain end : tf.memory()', tf.memory());
			tf.dispose(trainXs);
			tf.dispose(trainYs);
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
			callbacks: { onBatchBegin, onBatchEnd, onEpochEnd, onTrainEnd }
		};
		if (valXs && valYs) {
			params['validationData'] = [valXs, valYs];
		}

		return networkUnderTraining.tfModel.fit(trainXs, trainYs, params);
	}

	async function train10() {
		trainOnData(10, 10);
	}

	async function train100() {
		trainOnData(100, 50);
	}

	function resetModel() {
		networkStore.reload();
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
			<div class="stats stats-vertical shadow">
				<div class="stat">
					<div class="stat-title">Temperature</div>
					<input
						type="range"
						min={MIN_TEMP}
						max={MAX_TEMP}
						value={temperature}
						class="range range-primary"
						on:change={setTemperature}
					/>
					<div class="stat-value">{temperature} °C</div>
				</div>
				<div class="stat">
					<div class="stat-title">Humidité relative</div>
					<input
						type="range"
						min={MIN_HUMIDITY}
						max={MAX_HUMIDITY}
						value={relativeHumidity}
						class="range range-primary"
						on:change={setRelativeHumidity}
					/>
					<div class="stat-value">{relativeHumidity} %</div>
				</div>
				<div class="stat">
					<div class="stat-title">Index de chaleur prédit</div>
					<div class="stat-value">
						{formattedPrediction} °C
					</div>
				</div>
				<div class="stat">
					<div class="stat-title">Index de chaleur calculé</div>
					<div class="stat-value">
						{formattedComputedHeatIndex} °C
					</div>
				</div>
			</div>

			<div class="divider"></div>
			<h1 class="text-2xl mt-0 mb-2">Apprentissage</h1>

			<h4 class="text-xl mb-2">Taux d'apprentissage</h4>
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
					<button class="btn btn-outline btn-primary" on:click={train10}>
						Entraîner avec 10 exemples
					</button>
				</li>
			</ul>
			<ul class="menu py-4">
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train100}>
						Entraîner avec 100 exemples
					</button>
				</li>
			</ul>
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
