import { DenseNetwork } from './NetworkShape';

import * as tf from '@tensorflow/tfjs';

type TrainingRound = {
	samplesSeen: number;
	finalAccuracy?: number;
};

export type NetworkStats = {
	samplesSeen: number;
	accuracy?: number;
};

export class NetworkUnderTraining {
	readonly tfModel: tf.Sequential;
	readonly featureModel: tf.LayersModel;
	readonly shape: DenseNetwork;

	private _samplesSeen: number = 0;
	private trainingStats: TrainingRound[] = [];

	constructor(tfModel: tf.Sequential, networkShape: DenseNetwork) {
		this.tfModel = tfModel;
		this.featureModel = toFeatureModel(tfModel);
		this.shape = networkShape;
	}

	get stats(): NetworkStats {
		let accuracy;
		if (this.trainingStats.length > 0) {
			accuracy = this.trainingStats[this.trainingStats.length-1].finalAccuracy;
		}
		return {
			samplesSeen: this._samplesSeen,
			accuracy: accuracy
		};
	}

	trainingRoundDone(trainingRound: TrainingRound) {
		this.trainingStats.push(trainingRound);
		this._samplesSeen += trainingRound.samplesSeen;
	}
}

// The feature model lets us retrieve the activations of intermediate layers
function toFeatureModel(model: tf.Sequential): tf.LayersModel {
	const outputs = [];
	for (const layer of model.layers) {
		outputs.push(layer.output);
	}
	return tf.model({ inputs: model.input, outputs: outputs.flat() });
}

export function newAllDigitsNetwork(): NetworkUnderTraining {
	return new NetworkUnderTraining(newAllDigitsTFModel(), newAllDigitsNetworkShape());
}

function newAllDigitsTFModel(): tf.Sequential {
	const model = tf.sequential();
	model.add(tf.layers.dense({ inputShape: [28 * 28], units: 32, activation: 'relu' }));
	model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
	model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

	// This can be changed later, while training the model
	const optimizer = tf.train.adam();

	model.compile({
		optimizer: optimizer,
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy']
	});
	return model;
}

function newAllDigitsNetworkShape(): DenseNetwork {
	return new DenseNetwork(
		15,
		{
			name: "Couche d'entrée",
			neuron_count: 28 * 28,
			width: 22,
			height: 22,
			marker_size: 6,
			neurons_per_row: 28
		},
		{
			name: 'Couche cachée 1',
			neuron_count: 32,
			width: 60,
			height: 0,
			marker_size: 18
		},
		{
			name: 'Couche cachée 2',
			neuron_count: 32,
			width: 60,
			height: 0,
			marker_size: 18
		},
		{
			name: 'Couche de sortie',
			neuron_count: 10,
			width: 50,
			height: 5,
			marker_size: 18,
			labels: Array.from({ length: 10 }, (_, i) => i.toString())
		}
	);
}

export function newTwoDigitsNetwork(): NetworkUnderTraining {
	return new NetworkUnderTraining(newTwoDigitsTFModel(), newTwoDigitsNetworkShape());
}

function newTwoDigitsTFModel(): tf.Sequential {
	const model = tf.sequential();
	model.add(tf.layers.dense({ inputShape: [28 * 28], units: 32, activation: 'relu' }));
	model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
	model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

	// This can be changed later, while training the model
	const optimizer = tf.train.adam();

	model.compile({
		optimizer: optimizer,
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy']
	});
	return model;
}

function newTwoDigitsNetworkShape(): DenseNetwork {
	return new DenseNetwork(
		15,
		{
			name: "Couche d'entrée",
			neuron_count: 28 * 28,
			width: 22,
			height: 22,
			marker_size: 6,
			neurons_per_row: 28
		},
		{
			name: 'Couche cachée 1',
			neuron_count: 32,
			width: 60,
			height: 0,
			marker_size: 18
		},
		{
			name: 'Couche cachée 2',
			neuron_count: 16,
			width: 40,
			height: 0,
			marker_size: 18
		},
		{
			name: 'Couche de sortie',
			neuron_count: 2,
			width: 20,
			height: 5,
			marker_size: 18,
			labels: Array.from({ length: 2 }, (_, i) => i.toString())
		}
	);
}
