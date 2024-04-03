import { DenseNetwork } from './NetworkShape';

import * as tf from '@tensorflow/tfjs';
import {
	MnistData,
	allDigitsDataSourceConfig,
	zeroOneDataSourceConfig,
	type DataSource
} from './datasource';

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
	readonly dataSource: DataSource;

	private _samplesSeen: number = 0;
	private trainingStats: TrainingRound[] = [];

	constructor(tfModel: tf.Sequential, networkShape: DenseNetwork, dataSource: DataSource) {
		this.tfModel = tfModel;
		this.featureModel = toFeatureModel(tfModel);
		this.shape = networkShape;
		this.dataSource = dataSource;
	}

	nextTrainBatch(batchSize: number) {
		return this.dataSource.nextTrainBatch(batchSize);
	}

	nextTestBatch(batchSize: number) {
		return this.dataSource.nextTrainBatch(batchSize);
	}

	get stats(): NetworkStats {
		let accuracy;
		if (this.trainingStats.length > 0) {
			accuracy = this.trainingStats[this.trainingStats.length - 1].finalAccuracy;
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
	return new NetworkUnderTraining(
		newAllDigitsTFModel(),
		newAllDigitsNetworkShape(),
		new MnistData(allDigitsDataSourceConfig)
	);
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
			width: 24,
			height: 24,
			marker_size: 7,
			neurons_per_row: 28
		},
		{
			name: 'Couche cachée 1',
			neuron_count: 32,
			width: 55,
			height: 0,
			marker_size: 17
		},
		{
			name: 'Couche cachée 2',
			neuron_count: 32,
			width: 55,
			height: 0,
			marker_size: 17
		},
		{
			name: 'Couche de sortie',
			neuron_count: 10,
			width: 40,
			height: 5,
			marker_size: 17,
			labels: Array.from({ length: 10 }, (_, i) => i.toString())
		}
	);
}

export function newZeroOneNetwork(): NetworkUnderTraining {
	return new NetworkUnderTraining(
		newZeroOneTFModel(),
		newZeroOneNetworkShape(),
		new MnistData(zeroOneDataSourceConfig)
	);
}

function newZeroOneTFModel(): tf.Sequential {
	const model = tf.sequential();
	model.add(tf.layers.dense({ inputShape: [28 * 28], units: 32, activation: 'relu' }));
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

function newZeroOneNetworkShape(): DenseNetwork {
	return new DenseNetwork(
		12,
		{
			name: "Couche d'entrée",
			neuron_count: 28 * 28,
			width: 24,
			height: 24,
			marker_size: 7,
			neurons_per_row: 28
		},
		{
			name: 'Couche cachée 1',
			neuron_count: 32,
			width: 40,
			height: 0,
			marker_size: 14
		},
		{
			name: 'Couche de sortie',
			neuron_count: 2,
			width: 20,
			height: 0,
			marker_size: 17,
			labels: Array.from({ length: 2 }, (_, i) => i.toString())
		}
	);
}
