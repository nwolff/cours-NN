import { DenseNetwork } from '$lib/NetworkShape';
import { Network } from '$lib/Network';
import type { DataSource, DataBatch } from '$lib/DataSource';
import { zip2 } from '$lib/generic/utils';
import * as tf from '@tensorflow/tfjs';

export function newTemperatureControlNetwork(): Network {
	return new Network(
		newTemperatureControlTFModel(),
		newTemperatureControlNetworkShape(),
		new TemperatureControlDataSource()
	);
}

function newTemperatureControlTFModel(): tf.Sequential {
	const model = tf.sequential();
	model.add(tf.layers.dense({ inputShape: [2], units: 1, activation: 'relu' }));

	// This can be changed later, while training the model
	const optimizer = new tf.SGDOptimizer(0.1);

	model.compile({
		optimizer: optimizer,
		loss: tf.losses.absoluteDifference,
		metrics: tf.metrics.meanAbsoluteError
	});
	return model;
}

function newTemperatureControlNetworkShape(): DenseNetwork {
	return new DenseNetwork(
		3,
		{
			name: "Couche d'entrée",
			neuron_count: 2,
			width: 3,
			height: 0,
			marker_size: 20,
			classes: ['Température', 'Humidité relative']
		},
		{
			name: 'Couche de sortie',
			neuron_count: 1,
			width: 3,
			height: 0,
			marker_size: 20,
			classes: ['Index de chaleur']
		}
	);
}

/**
 * https://en.wikipedia.org/wiki/Heat_index
 *
 * @param t temperature in Celsius
 * @param r relative humidity between 0 and 100
 * @returns the heat index in Celsius
 */
export function computeHeatIndex(t: number, r: number): number {
	return (
		-8.784 +
		1.611 * t +
		2.338 * r +
		-0.1461 * t * r +
		-0.0123 * t ** 2 +
		-0.01642 * r ** 2 +
		2.211e-3 * t ** 2 * r +
		7.254e-4 * t * r ** 2 +
		-3.582e-6 * t ** 2 * r ** 2
	);
}

export const MIN_HUMIDITY = 40;
export const MAX_HUMIDITY = 100;

export function humidityToActivation(humidity: number): number {
	return (humidity - MIN_HUMIDITY) / (MAX_HUMIDITY - MIN_HUMIDITY);
}

export function activationToHumidity(activation: number): number {
	return MIN_HUMIDITY + activation * (MAX_HUMIDITY - MIN_HUMIDITY);
}

export const MIN_TEMP = 27;

export const MAX_TEMP = 44;

export function tempToActivation(temp: number): number {
	return (temp - MIN_TEMP) / (MAX_TEMP - MIN_TEMP);
}

export function activationToTemp(activation: number): number {
	return MIN_TEMP + activation * (MAX_TEMP - MIN_TEMP);
}

export const MIN_HEAT_INDEX = MIN_TEMP;
export const MAX_HEAT_INDEX = 65; // XXX: Calculation does not agree

export function heatIndexToActivation(heatIndex: number): number {
	return (heatIndex - MIN_HEAT_INDEX) / (MAX_HEAT_INDEX - MIN_HEAT_INDEX);
}

export function activationToHeatIndex(activation: number): number {
	return MIN_HEAT_INDEX + activation * (MAX_HEAT_INDEX - MIN_HEAT_INDEX);
}

class TemperatureControlDataSource implements DataSource {
	async loadData(): Promise<void> {}

	nextTrainBatch(batchSize: number): DataBatch {
		// XXX: This leaks
		const temperatureTensor: tf.Tensor1D = tf.randomUniform([batchSize], 0, 1);
		const humidityTensor: tf.Tensor1D = tf.randomUniform([batchSize], 0, 1);
		const temperatures = temperatureTensor.arraySync();
		const humidities = humidityTensor.arraySync();

		const heatIndices = [];
		for (const [temp, humidity] of zip2(temperatures, humidities)) {
			const heatIndex = computeHeatIndex(activationToTemp(temp), activationToHumidity(humidity));
			heatIndices.push([heatIndexToActivation(heatIndex)]);
		}

		return {
			xs: tf.stack([temperatureTensor, humidityTensor]).reshape([batchSize, -1]) as tf.Tensor2D,
			ys: tf.tensor(heatIndices)
		};
	}

	nextTestBatch(batchSize: number): DataBatch {
		return this.nextTrainBatch(batchSize);
	}
}
