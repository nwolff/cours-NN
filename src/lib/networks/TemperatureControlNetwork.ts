import { DenseNetwork } from '$lib/NetworkShape';
import { Network } from '$lib/Network';
import type { DataSource, DataBatch } from '$lib/DataSource';
import { zip3 } from '$lib/generic/utils';
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
	model.add(
		tf.layers.dense({
			inputShape: [3],
			units: 1,
			activation: 'linear'
			// activation: 'sigmoid',
			// biasInitializer: 'randomNormal'
		})
	);
	/*
	// This can be changed later, while training the model
	const optimizer = 'adam'; // new tf.SGDOptimizer(0.1);

	model.compile({
		optimizer: optimizer,
		loss: 'meanSquaredError',
		metrics: ['accuracy']
	});
	*/
	// This can be changed later, while training the model
	// const optimizer = new tf.SGDOptimizer(0.1);

	/*
	model.compile({
		optimizer: optimizer,
		loss: tf.losses.absoluteDifference,
		metrics: tf.metrics.meanAbsoluteError
	});
	*/

	/*
	const newWeights = tf.tensor2d([[1.0], [-0.19], [0.33]], [3, 1]);
	const newBiases = tf.tensor1d([-4]);

	model.setWeights([newWeights, newBiases]);

	// Cleanup: The model now has its own internal copies
	newWeights.dispose();
	newBiases.dispose();
	*/

	// 1. Get the array of weights for the first layer
	const layerParameters = model.layers[0].getWeights();

	// 2. Separate weights and biases
	const weights = layerParameters[0];
	const biases = layerParameters[1];

	// 3. View them in the console
	console.log('--- Weights (Kernel) ---');
	weights.print();

	console.log('--- Biases ---');
	biases.print();

	// This can be changed later, while training the model
	const optimizer = new tf.SGDOptimizer(0.01);
	model.compile({
		optimizer: 'adam',
		// optimizer: optimizer,
		loss: 'meanSquaredError'
	});
	return model;
}

function newTemperatureControlNetworkShape(): DenseNetwork {
	return new DenseNetwork(
		3,
		{
			name: "Couche d'entrée",
			neuron_count: 3,
			width: 3,
			height: 0,
			marker_size: 20,
			classes: ['Température', 'Vent', "Vapeur d'eau"]
		},
		{
			name: 'Couche de sortie',
			neuron_count: 1,
			width: 3,
			height: 0,
			marker_size: 20,
			classes: ['Temperature apparente (°C)']
		}
	);
}

/**
 * Australian Apparent Temperature (AT)
 * Robert Steadman
 *
 * https://journals.ametsoc.org/view/journals/apme/23/12/1520-0450_1984_023_1674_ausoat_2_0_co_2.pdf
 *
 * The formula is AT = (1.0 * T) + (0.33 * e) - (0.70 * v) - 4.0
 * I chose it because it is a linear combination of the inputs, and a single neuron can compute it.
 *
 * Ideally would have preferred to use relative humidity which is intuitive instead of e,
 * but that gives us a nonlinear equation. So we stick to e in hectopascal.
 *
 * This table shows the value of e for different temperature / relative humidity conditions:
 *
 * Temp (°C)	20% RH	40% RH	60% RH	80% RH	100% RH
 * 	    0°C		1.22	2.44	3.66	4.88	6.10
 * 	   10°C		2.45	4.90	7.36	9.81	12.26
 * 	   20°C		4.66	9.33	13.99	18.66	23.32
 * 	   30°C		8.46	16.91	25.37	33.83	42.29
 *
 * @param T_C temperature in Celsius
 * @param e_hPa Water Vapor Pressure in hPa
 * @param v_ms wind speed in m/s
 * @returns the apparent temperature in Celsius
 */
export function computeApparentTemperature(T_C: number, v_kph: number, e_hPa: number): number {
	// -0.19 = -0.7 * 1000 / 3600
	return 1.0 * T_C - 0.19 * v_kph + 0.33 * e_hPa - 4.0;
}

export const MIN_TEMPERATURE = 0;
export const MAX_TEMPERATURE = 30;

const id = (x) => x;

export const temperatureToActivation = id;
export const activationToTemperature = id;
export const windSpeedToActivation = id;
export const activationToWindSpeed = id;
export const waterVaporPressureToActivation = id;
export const activationToWaterVaporPressure = id;
export const apparentTemperatureToActivation = id;
export const activationToApparentTemperature = id;
/*
export function temperatureToActivation(temp: number): number {
	return (temp - MIN_TEMPERATURE) / (MAX_TEMPERATURE - MIN_TEMPERATURE);
}

export function activationToTemperature(activation: number): number {
	return MIN_TEMPERATURE + activation * (MAX_TEMPERATURE - MIN_TEMPERATURE);
}
*/

export const MIN_WIND_SPEED = 0;
export const MAX_WIND_SPEED = 100;

/*
export function windSpeedToActivation(windSpeed: number): number {
	return (windSpeed - MIN_WIND_SPEED) / (MAX_WIND_SPEED - MIN_WIND_SPEED);
}

export function activationToWindSpeed(activation: number): number {
	return MIN_WIND_SPEED + activation * (MAX_WIND_SPEED - MIN_WIND_SPEED);
}
*/

export const MIN_WATER_VAPOR_PRESSURE = 1.22;
export const MAX_WATER_VAPORT_PRESSURE = 43;

/*
export function waterVaporPressureToActivation(waterVaporPressure: number): number {
	return (
		(waterVaporPressure - MIN_WATER_VAPOR_PRESSURE) /
		(MAX_WATER_VAPORT_PRESSURE - MIN_WATER_VAPOR_PRESSURE)
	);
}

export function activationToWaterVaporPressure(activation: number): number {
	return (
		MIN_WATER_VAPOR_PRESSURE + activation * (MAX_WATER_VAPORT_PRESSURE - MIN_WATER_VAPOR_PRESSURE)
	);
}
*/

/*
export const MIN_APPARENT_TEMPERATURE = -3.6;
export const MAX_APPARENT_TEMPERATURE = 40;

export function apparentTemperatureToActivation(apparentTemperature: number): number {
	return (
		(apparentTemperature - MIN_APPARENT_TEMPERATURE) /
		(MAX_APPARENT_TEMPERATURE - MIN_APPARENT_TEMPERATURE)
	);
}


export function activationToApparentTemperature(activation: number): number {
	return (
		MIN_APPARENT_TEMPERATURE + activation * (MAX_APPARENT_TEMPERATURE - MIN_APPARENT_TEMPERATURE)
	);
}
*/

class TemperatureControlDataSource implements DataSource {
	async loadData(): Promise<void> {}

	nextTrainBatch(batchSize: number): DataBatch {
		// XXX: This leaks
		// 1. Generate random input inside the range
		const temperature: tf.Tensor2D = tf.randomUniform(
			[batchSize, 1],
			MIN_TEMPERATURE,
			MAX_TEMPERATURE
		);

		const windSpeed: tf.Tensor2D = tf.randomUniform([batchSize, 1], MIN_WIND_SPEED, MAX_WIND_SPEED);

		const waterVaporPressure: tf.Tensor2D = tf.randomUniform(
			[batchSize, 1],
			MIN_WATER_VAPOR_PRESSURE,
			MAX_WATER_VAPORT_PRESSURE
		);

		const xs = tf.concat([temperature, windSpeed, waterVaporPressure], 1);

		// 2. Calculate y using your formula: 1.0*TC - 0.19*v + 0.33*e - 4.0
		// We add a tiny bit of noise (+/- 0.01)
		const ys: tf.Tensor2D = temperature
			.mul(1.0)
			.sub(windSpeed.mul(0.19))
			.add(waterVaporPressure.mul(0.33))
			.sub(4.0)
			.add(tf.randomNormal([batchSize, 1], 0, 0.01));

		return {
			xs: xs,
			ys: ys
		};
	}

	nextTestBatch(batchSize: number): DataBatch {
		return this.nextTrainBatch(batchSize);
	}
}

/**
 * // 1. Setup Data
const xs = tf.tensor2d([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
const ys = tf.tensor2d([[0.5], [0.8]]);

// 2. Setup Optimizer
const optimizer = tf.train.adam(0.01); // Using a specific training constant

// 3. Manual Training Loop
async function trainManual(epochs) {
  for (let i = 0; i < epochs; i++) {
    
    // We use a "Gradient Tape" style approach via optimizer.minimize()
    const cost = optimizer.minimize(() => {
      const pred = model.predict(xs);
      
      return tf.losses.meanSquaredError(ys, pred);
    }, true); // 'true' returns the cost tensor

    if (i % 10 === 0) {
      console.log(`Epoch ${i}: Loss = ${cost.dataSync()}`);
    }
    
    // Clean up memory
    tf.dispose(cost);
    await tf.nextFrame(); // Keep UI responsive
  }
}

trainManual(100);
 */
