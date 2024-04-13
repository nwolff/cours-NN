<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { temperatureControlNetworkStore } from '../../stores';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
	import { testNetwork } from '$lib/NetworkTesting';
	import { allLinks } from '$lib/LinkFilters';
	import {
		activationToApparentTemperature,
		computeApparentTemperature,
		MAX_TEMPERATURE,
		MAX_WATER_VAPORT_PRESSURE,
		MAX_WIND_SPEED,
		MIN_TEMPERATURE,
		MIN_WATER_VAPOR_PRESSURE,
		MIN_WIND_SPEED,
		temperatureToActivation,
		waterVaporPressureToActivation,
		windSpeedToActivation
	} from '$lib/networks/TemperatureControlNetwork';

	const logger = new tslog.Logger({ name: 'temperature' });

	const networkStore = temperatureControlNetworkStore;

	const LEARNING_RATE = 0.2; // XXX: made this up

	$: networkShape = $networkStore?.shape;
	$: classes = $networkStore?.shape.classes;
	$: weights = $networkStore?.tfModel.weights;

	$: temperature = MIN_TEMPERATURE;
	$: windSpeed = MIN_WIND_SPEED;
	$: waterVaporPressure = MIN_WATER_VAPOR_PRESSURE;

	let prediction: number[] | undefined;
	let activations: number[][] | undefined;
	let linkFilter = allLinks;

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	$: formattedPrediction = formatter.format(activationToApparentTemperature(prediction));

	$: formattedComputedApparentTemperature = formatter.format(
		computeApparentTemperature(temperature, windSpeed, waterVaporPressure)
	);

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		isLoading = false;
	});

	function setTemperature(event: Event) {
		temperature = event.target?.valueAsNumber;
		predict_apparent_temperature();
	}

	function setWindSpeed(event: Event) {
		windSpeed = event.target?.valueAsNumber;
		predict_apparent_temperature();
	}

	function setWaterVaporPressure(event: Event) {
		waterVaporPressure = event.target?.valueAsNumber;
		predict_apparent_temperature();
	}

	function predict_apparent_temperature() {
		activations = calculateActivations(temperature, windSpeed, waterVaporPressure);
		prediction = activations[activations.length - 1];
		logger.info('prediction:', prediction); // XXX
		logger.debug('tf.memory() ', tf.memory());
	}

	function calculateActivations(
		temperature: number,
		windSpeed: number,
		waterVaporPressure: number
	): number[][] {
		return tf.tidy(() => {
			let input = tf
				.tensor([
					temperatureToActivation(temperature),
					windSpeedToActivation(windSpeed),
					waterVaporPressureToActivation(waterVaporPressure)
				])
				.reshape([-1, 3]);
			const activationTensor = $networkStore.featureModel.predict(input) as tf.Tensor[];
			return [input, activationTensor].map((x) => tf.squeeze(x).arraySync()) as number[][];
		});
	}

	async function trainOnData(trainDataSize: number) {
		const trainData = $networkStore.nextTrainBatch(trainDataSize);
		trainData.xs.print(); // XXX
		trainData.ys.print(); // XXX
		train(trainData.xs, trainData.ys, null, null, 1, LEARNING_RATE);
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
		// const optimizer = networkUnderTraining.tfModel.optimizer as tf.SGDOptimizer;
		// optimizer.setLearningRate(learningRate);

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
			// tf.dispose(trainYs); // For this network they live forever. XXX
			if (valXs) {
				tf.dispose(valXs);
			}
			if (valYs) {
				tf.dispose(valYs);
			}
			logger.debug('after disposing: tf.memory()', tf.memory());

			predict_apparent_temperature();
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

	async function train1() {
		trainOnData(1);
	}

	async function train10() {
		trainOnData(10);
	}

	function resetModel() {
		networkStore.reload();
		predict_apparent_temperature();
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-9 gap-4">
		<div class="col-span-2">
			<div class="stats stats-vertical shadow">
				<div class="stat">
					<div class="stat-title">Température</div>
					<input
						type="range"
						min={MIN_TEMPERATURE}
						max={MAX_TEMPERATURE}
						value={temperature}
						class="range range-primary"
						on:change={setTemperature}
					/>
					<div class="stat-value">{temperature} °C</div>
				</div>
				<div class="stat">
					<div class="stat-title">Vent</div>
					<input
						type="range"
						min={MIN_WIND_SPEED}
						max={MAX_WIND_SPEED}
						value={windSpeed}
						class="range range-primary"
						on:change={setWindSpeed}
					/>
					<div class="stat-value">{windSpeed} km/h</div>
				</div>
				<div class="stat">
					<div class="stat-title">Vapeur d'eau</div>
					<input
						type="range"
						min={MIN_WATER_VAPOR_PRESSURE}
						max={MAX_WATER_VAPORT_PRESSURE}
						value={waterVaporPressure}
						class="range range-primary"
						on:change={setWaterVaporPressure}
					/>
					<div class="stat-value">{waterVaporPressure} hPa</div>
				</div>
				<div class="stat">
					<div class="stat-title">Temperature apparente calculée</div>
					<div class="stat-value">
						{formattedComputedApparentTemperature} °C
					</div>
				</div>
				<div class="stat">
					<div class="stat-title">Température apparente prédite</div>
					<div class="stat-value">
						{formattedPrediction} °C
					</div>
				</div>
			</div>

			<div class="divider"></div>
			<h1 class="text-2xl mt-0 mb-2">Apprentissage</h1>

			<ul class="menu py-4">
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train1}>
						Entraîner avec 1 exemple
					</button>
				</li>
			</ul>
			<ul class="menu py-4">
				<li class="mt-1">
					<button class="btn btn-outline btn-primary" on:click={train10}>
						Entraîner avec 10 exemples
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
