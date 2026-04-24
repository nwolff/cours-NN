<script lang="ts">
	import { onMount } from 'svelte';
	import * as tf from '@tensorflow/tfjs';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { temperatureControlNetworkStore } from '../../stores';
	import * as tslog from 'tslog';
	import { allLinks } from '$lib/LinkFilters';
	import NetworkStats from '$lib/components/NetworkStats.svelte';
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

	let _tick = $state(0);

	const networkShape = $derived($networkStore?.shape);
	const weights = $derived.by(() => {
		void _tick;
		const w = $networkStore?.tfModel.weights;
		return w ? [...w] : undefined;
	});
	const neuronFormula = $derived(buildNeuronFormula(weights));
	const currentStats = $derived.by(() => {
		void _tick;
		const s = $networkStore?.stats;
		return s ? { ...s, losses: [...s.losses] } : null;
	});

	const formulaFormatter = Intl.NumberFormat('en', {
		notation: 'compact',
		maximumFractionDigits: 2
	});

	function buildNeuronFormula(tfWeights: tf.LayerVariable[] | undefined) {
		// Build something like:  1,0 * T - 0.19 * V + 0.33 * H - 4
		if (!tfWeights) {
			return;
		}
		const withOperator = (x: number) =>
			x > 0 ? `+ ${formulaFormatter.format(x)}` : `- ${formulaFormatter.format(Math.abs(x))}`;
		const [weights, biases] = tfWeights.map((w) => w.read().flatten().arraySync() as number[]);
		const [wT, wV, wH] = weights;
		const bias = biases[0];
		return `${formulaFormatter.format(wT)} * T ${withOperator(wV)} * V ${withOperator(wH)} * H ${withOperator(bias)}`;
	}

	const formatter = Intl.NumberFormat('en', { notation: 'compact' });

	let temperature = $state(MIN_TEMPERATURE);
	let windSpeed = $state(MIN_WIND_SPEED);
	let waterVaporPressure = $state(MIN_WATER_VAPOR_PRESSURE);

	let prediction: number[] | undefined = $state(undefined);
	let activations: number[][] | undefined = $state(undefined);
	let linkFilter = allLinks;

	const formattedPrediction = $derived(formatter.format(prediction as any));
	const formattedComputedApparentTemperature = $derived(
		formatter.format(computeApparentTemperature(temperature, windSpeed, waterVaporPressure))
	);

	let isLoading = $state(true);

	onMount(async () => {
		await networkStore.load();
		isLoading = false;
		// setApparentTemperatureFunctionWeights($networkStore.tfModel);
		predict_apparent_temperature();
	});

	const randomUniformInt = (min: number, max: number) =>
		Math.floor(Math.random() * (max - min) + min);

	function show_example_in_UI() {
		temperature = randomUniformInt(MIN_TEMPERATURE, MAX_TEMPERATURE);
		windSpeed = randomUniformInt(MIN_WIND_SPEED, MAX_WIND_SPEED);
		waterVaporPressure = randomUniformInt(MIN_WATER_VAPOR_PRESSURE, MAX_WATER_VAPORT_PRESSURE);
		predict_apparent_temperature();
	}

	function predict_apparent_temperature() {
		const result = calculateActivations(temperature, windSpeed, waterVaporPressure);
		activations = result;
		prediction = result[result.length - 1];
		logger.debug('tf.memory() ', tf.memory());
	}

	function calculateActivations(
		temperature: number,
		windSpeed: number,
		waterVaporPressure: number
	): number[][] {
		return tf.tidy(() => {
			const input = tf.tensor([temperature, windSpeed, waterVaporPressure]).reshape([-1, 3]);
			const rawPrediction = $networkStore.featureModel.predict(input);
			// predict() returns Tensor when there is a single output layer, Tensor[] otherwise
			const activationTensors: tf.Tensor[] = Array.isArray(rawPrediction)
				? rawPrediction
				: [rawPrediction as tf.Tensor];
			const layerActivations = activationTensors.map((x) => {
				const values = tf.squeeze(x).arraySync();
				// A single-unit layer produces a scalar; wrap it so every layer has number[]
				return (Array.isArray(values) ? values : [values]) as number[];
			});
			const inputActivations = tf.squeeze(input).arraySync() as number[];
			return [inputActivations, ...layerActivations];
		});
	}

	async function trainOnData(trainDataSize: number) {
		const trainData = $networkStore.nextTrainBatch(trainDataSize);
		train(trainData.xs, trainData.ys, 32);
	}

	async function train(trainXs: tf.Tensor2D, trainYs: tf.Tensor2D, batchSize: number) {
		const networkUnderTraining = $networkStore;

		function onBatchEnd(batch: number, logs: tf.Logs = {}) {
			logger.debug('end batch:', batch, '. logs:', logs);
			networkUnderTraining.trainingRoundDone({
				samplesSeen: logs.size,
				finalAccuracy: logs.acc,
				loss: logs.loss
			});
			_tick++; // Notify subscribers
			show_example_in_UI();
		}

		function onEpochEnd(epoch: number, logs: tf.Logs = {}) {
			logger.debug('end epoch:', epoch, '. logs:', logs);
		}

		function onTrainEnd(_logs: tf.Logs = {}) {
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
		const params: tf.ModelFitArgs = {
			epochs: 1,
			batchSize: batchSize,
			shuffle: false,
			callbacks: { onBatchEnd, onEpochEnd, onTrainEnd } as tf.CustomCallbackArgs
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
		_tick++; // Notify subscribers
		predict_apparent_temperature();
	}

	function setATFunctionWeights() {
		const networkUnderTraining = $networkStore;
		setApparentTemperatureFunctionWeights(networkUnderTraining.tfModel);
		_tick++; // Notify subscribers
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
					<div class="stat-title !text-base">Calcul</div>
					<div class="stat-value !text-xl">1 * T - 0.19 * V + 0.33 * H - 4</div>
					<div class="stat-title !text-base">Température apparente</div>
					<div class="stat-value !text-xl">
						{formattedComputedApparentTemperature} °C
					</div>
				</div>
			</div>
			<div class="divider mt-10"></div>
			<h1 class="text-2xl mt-3 mb-6">Apprentissage</h1>

			<button class="btn btn-outline btn-primary" onclick={train1}>
				Entraîner avec 1 exemple
			</button>
			<div class="m-6"></div>
			<button class="btn btn-outline btn-primary" onclick={train100}>
				Entraîner avec 100 exemples
			</button>
			<div class="m-6"></div>
			<button class="btn btn-outline btn-primary" onclick={train1000}>
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
							<div class="stat-title !text-base">Température</div>
							<input
								type="range"
								min={MIN_TEMPERATURE}
								max={MAX_TEMPERATURE}
								bind:value={temperature}
								oninput={predict_apparent_temperature}
								class="range range-primary range-xs"
							/>
							<div class="stat-value !text-xl">{temperature} °C</div>
						</div>
						<div class="stat">
							<div class="stat-title !text-base">Vent</div>
							<input
								type="range"
								min={MIN_WIND_SPEED}
								max={MAX_WIND_SPEED}
								bind:value={windSpeed}
								oninput={predict_apparent_temperature}
								class="range range-primary range-xs"
							/>
							<div class="stat-value !text-xl">{windSpeed} km/h</div>
						</div>
						<div class="stat">
							<div class="stat-title !text-base">Humidité</div>
							<input
								type="range"
								min={MIN_WATER_VAPOR_PRESSURE}
								max={MAX_WATER_VAPORT_PRESSURE}
								bind:value={waterVaporPressure}
								oninput={predict_apparent_temperature}
								class="range range-primary range-xs"
							/>
							<div class="stat-value !text-xl">{waterVaporPressure} hPa</div>
						</div>
					</div>
				</div>
				<!-- Network -->
				<div class="grid p-10">
					<NetworkGraph
						{networkShape}
						weights={weights ?? []}
						activations={activations ?? []}
						{linkFilter}
						style=""
					/>
				</div>
				<!-- Output -->
				<div class="grid place-items-center">
					<div class="stats stats-vertical shadow">
						<div class="stat">
							<div class="stat-title !text-base">Calcul</div>
							<div class="stat-value !text-xl">{neuronFormula}</div>
							<div class="stat-title !text-base">Température apparente</div>
							<div class="stat-value !text-xl">
								{formattedPrediction} °C
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>

		<!-- Right -->
		<div class="col-span-2">
			<NetworkStats stats={currentStats!} />
			<div class="m-6"></div>
			<button class="btn btn-outline btn-error" onclick={resetModel}>
				Réinitialiser le réseau
			</button>
			<div class="m-6"></div>
			<button class="btn btn-outline btn-primary" onclick={setATFunctionWeights}>
				Réseau cible
			</button>
		</div>
	</div>
{/if}
