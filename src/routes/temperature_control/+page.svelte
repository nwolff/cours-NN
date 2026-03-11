<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { temperatureControlNetworkStore } from '../../stores';
	import * as tslog from 'tslog';
	import { allLinks } from '$lib/LinkFilters';
	import LossChart from '$lib/components/LossChart.svelte';
	import {
		computeApparentTemperature,
		MAX_TEMPERATURE,
		MAX_WATER_VAPORT_PRESSURE,
		MAX_WIND_SPEED,
		MIN_TEMPERATURE,
		MIN_WATER_VAPOR_PRESSURE,
		MIN_WIND_SPEED,
		setApparentTemperatureFunctionWeights
	} from '$lib/networks/TemperatureControlNetwork';

	const logger = new tslog.Logger({ name: 'temperature' });

	const networkStore = temperatureControlNetworkStore;

	$: networkShape = $networkStore?.shape;
	$: weights = $networkStore?.tfModel.weights;
	$: neuronFormula = buildNeuronFormula(weights);

	const formulaFormatter = Intl.NumberFormat('en', {
		notation: 'compact',
		maximumFractionDigits: 2
	});

	function buildNeuronFormula(tfWeights) {
		// Build something like:  1,0 * T - 0.19 * V + 0.33 * H - 4
		if (!tfWeights) {
			return;
		}
		const withOperator = (x) => (x > 0 ? `+ ${x}` : `- ${Math.abs(x)}`);
		const [weights, biases] = tfWeights.map((w) => w.read().flatten().arraySync());
		const [wT, wV, wH] = weights.map((w) => formulaFormatter.format(w));
		const bias = formulaFormatter.format(biases[0]);
		return `${wT} * T ${withOperator(wV)} * V ${withOperator(wH)} * H ${withOperator(bias)}`;
	}

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	$: temperature = MIN_TEMPERATURE;
	$: windSpeed = MIN_WIND_SPEED;
	$: waterVaporPressure = MIN_WATER_VAPOR_PRESSURE;

	let prediction: number[] | undefined;
	let activations: number[][] | undefined;
	let linkFilter = allLinks;

	$: formattedPrediction = formatter.format(prediction);

	$: formattedComputedApparentTemperature = formatter.format(
		computeApparentTemperature(temperature, windSpeed, waterVaporPressure)
	);

	// Whenever the user moves the slider, we want to predict.
	// The first three lines make the svelte compiler understand that changes
	// to those variables need to trigger this block
	$: {
		temperature;
		windSpeed;
		waterVaporPressure;
		if (!isLoading) {
			predict_apparent_temperature();
			prediction = prediction; // To trigger a reactive chain
		}
	}

	$: stats = $networkStore?.stats;
	$: formattedNumExamples = formatter.format(stats?.samplesSeen);
	$: losses = stats?.losses;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		isLoading = false;
		// setApparentTemperatureFunctionWeights($networkStore.tfModel);
		predict_apparent_temperature();
	});

	const randomUniformInt = (min, max) => Math.floor(Math.random() * (max - min) + min);

	function show_example_in_UI() {
		temperature = randomUniformInt(MIN_TEMPERATURE, MAX_TEMPERATURE);
		windSpeed = randomUniformInt(MIN_WIND_SPEED, MAX_WIND_SPEED);
		waterVaporPressure = randomUniformInt(MIN_WATER_VAPOR_PRESSURE, MAX_WATER_VAPORT_PRESSURE);
		predict_apparent_temperature();
	}

	function predict_apparent_temperature() {
		activations = calculateActivations(temperature, windSpeed, waterVaporPressure);
		prediction = activations[activations.length - 1];
		logger.debug('tf.memory() ', tf.memory());
	}

	function calculateActivations(
		temperature: number,
		windSpeed: number,
		waterVaporPressure: number
	): number[][] {
		return tf.tidy(() => {
			let input = tf.tensor([temperature, windSpeed, waterVaporPressure]).reshape([-1, 3]);
			const activationTensor = $networkStore.featureModel.predict(input) as tf.Tensor[];
			return [input, activationTensor].map((x) => tf.squeeze(x).arraySync()) as number[][];
		});
	}

	async function trainOnData(trainDataSize: number) {
		const trainData = $networkStore.nextTrainBatch(trainDataSize);
		train(trainData.xs, trainData.ys, 32);
	}

	async function train(trainXs: tf.Tensor2D, trainYs: tf.Tensor2D, batchSize: number) {
		const networkUnderTraining = $networkStore;

		function onBatchEnd(batch: number, logs: tf.Logs) {
			logger.debug('end batch:', batch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: logs.size,
				finalAccuracy: logs.acc,
				loss: logs.loss
			});
			networkStore.update((n) => n); // Notify subscribers
			show_example_in_UI();
		}

		function onEpochEnd(epoch: number, logs: tf.Logs) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
		}

		function onTrainEnd(_logs: tf.Logs) {
			logger.debug('onTrain end : tf.memory()', tf.memory());
			// tf.dispose(trainXs);
			// tf.dispose(trainYs); // For this network they live forever. XXX
			logger.debug('after disposing: tf.memory()', tf.memory());

			predict_apparent_temperature();
		}

		logger.debug('Before fit: tf.memory()', tf.memory());

		// If this fails because there is already another fit running
		// Then the 2 tensors get leaked (because the cleanup occurs in
		// onTrainEnd, which is never called)
		const params = {
			epochs: 1,
			batchSize: batchSize,
			shuffle: false,
			callbacks: { onBatchEnd, onEpochEnd, onTrainEnd }
		};
		return networkUnderTraining.tfModel.fit(trainXs, trainYs, params);
	}

	async function train1() {
		trainOnData(1);
	}

	async function train100() {
		trainOnData(100);
	}

	async function train1000() {
		trainOnData(1000);
	}

	function resetModel() {
		networkStore.reload();
		predict_apparent_temperature();
	}

	function setATFunctionWeights() {
		const networkUnderTraining = $networkStore;
		setApparentTemperatureFunctionWeights(networkUnderTraining.tfModel);
		networkStore.update((n) => n); // Notify subscribers
		predict_apparent_temperature();
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-8 gap-9">
		<!-- Left -->
		<div class="col-span-2">
			<h1 class="text-2xl mt-5 mb-2">Fonction Cible</h1>
			<div class="stats stats-vertical shadow">
				<div class="stat px-0">
					<div class="stat-title">Calcul</div>
					<div class="stat-value text-xl">1 * T - 0.19 * V + 0.33 * H - 4</div>
					<div class="stat-title">Température apparente</div>
					<div class="stat-value text-xl">
						{formattedComputedApparentTemperature} °C
					</div>
				</div>
			</div>
			<div class="divider mt-10"></div>
			<h1 class="text-2xl mt-3 mb-6">Apprentissage</h1>

			<button class="btn btn-outline btn-primary" on:click={train1}>
				Entraîner avec 1 exemple
			</button>
			<div class="m-6" />
			<button class="btn btn-outline btn-primary" on:click={train100}>
				Entraîner avec 100 exemples
			</button>
			<div class="m-6" />
			<button class="btn btn-outline btn-primary" on:click={train1000}>
				Entraîner avec 1000 exemples
			</button>
		</div>

		<!-- Center -->
		<div class="col-span-4">
			<div class="flex flex-col w-full">
				<!-- Controls -->
				<div class="grid h-20 place-items-center">
					<div class="stats stats-horizontal shadow">
						<div class="stat">
							<div class="stat-title">Température</div>
							<input
								type="range"
								min={MIN_TEMPERATURE}
								max={MAX_TEMPERATURE}
								bind:value={temperature}
								class="range range-primary range-xs"
							/>
							<div class="stat-value text-xl">{temperature} °C</div>
						</div>
						<div class="stat">
							<div class="stat-title">Vent</div>
							<input
								type="range"
								min={MIN_WIND_SPEED}
								max={MAX_WIND_SPEED}
								bind:value={windSpeed}
								class="range range-primary range-xs"
							/>
							<div class="stat-value text-xl">{windSpeed} km/h</div>
						</div>
						<div class="stat">
							<div class="stat-title">Humidité</div>
							<input
								type="range"
								min={MIN_WATER_VAPOR_PRESSURE}
								max={MAX_WATER_VAPORT_PRESSURE}
								bind:value={waterVaporPressure}
								class="range range-primary range-xs"
							/>
							<div class="stat-value text-xl">{waterVaporPressure} hPa</div>
						</div>
					</div>
				</div>
				<!-- Network -->
				<div class="grid p-10">
					<NetworkGraph {networkShape} {weights} {activations} {linkFilter} style="" />
				</div>
				<!-- Output -->
				<div class="grid place-items-center">
					<div class="stats stats-vertical shadow">
						<div class="stat">
							<div class="stat-title">Calcul</div>
							<div class="stat-value text-xl">{neuronFormula}</div>
							<div class="stat-title">Température apparente</div>
							<div class="stat-value text-xl">
								{formattedPrediction} °C
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>

		<!-- Right -->
		<div class="col-span-2">
			<div class="stats shadow bg-base-200 stats-vertical">
				<div class="stat">
					<div class="stat-title">Exemples vus</div>
					<div class="stat-value">{formattedNumExamples}</div>
				</div>
				<div class="stat">
					<div class="stat-title">Perte</div>
					<LossChart {losses} />
				</div>
			</div>
			<div class="m-6" />
			<button class="btn btn-outline btn-error" on:click={resetModel}>
				Réinitialiser le réseau
			</button>
			<div class="m-6" />
			<button class="btn btn-outline btn-primary" on:click={setATFunctionWeights}>
				Réseau cible
			</button>
		</div>
	</div>
{/if}
