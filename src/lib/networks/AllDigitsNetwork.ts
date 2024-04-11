import { DenseNetwork } from '$lib/NetworkShape';
import { Network } from '$lib/Network';
import * as tf from '@tensorflow/tfjs';
import { MnistData, type DataSourceConfig } from '$lib/DataSource';

export function newAllDigitsNetwork(): Network {
	return new Network(
		newAllDigitsTFModel(),
		newAllDigitsNetworkShape(),
		new MnistData(allDigitsDataSourceConfig)
	);
}

const allDigitsDataSourceConfig: DataSourceConfig = {
	imagesSpriteURL: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
	labelsURL: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
	imageSize: 28 * 28,
	numClasses: 10,
	numDatasetElements: 65_000,
	numTrainElements: 55_000
};

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
