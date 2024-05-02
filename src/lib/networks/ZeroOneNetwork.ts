import { DenseNetwork } from '$lib/NetworkShape';
import { Network } from '$lib/Network';
import { MnistData, type DataSourceConfig } from '$lib/DataSource';
import * as tf from '@tensorflow/tfjs';
import { base } from '$app/paths';

export function newZeroOneNetwork(): Network {
	return new Network(
		newZeroOneTFModel(),
		newZeroOneNetworkShape(),
		new MnistData(zeroOneDataSourceConfig)
	);
}

const zeroOneDataSourceConfig: DataSourceConfig = {
	imagesSpriteURL: base + '/zero_one_images.png',
	labelsURL: base + '/zero_one_labels_uint8',
	imageSize: 28 * 28,
	numClasses: 2,
	numDatasetElements: 14_780,
	numTrainElements: 12_000
};

function newZeroOneTFModel(): tf.Sequential {
	const model = tf.sequential();
	model.add(tf.layers.dense({ inputShape: [28 * 28], units: 16, activation: 'relu' }));
	model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));

	// This can be changed later, while training the model
	const optimizer = new tf.SGDOptimizer(0.1);

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
			neuron_count: 16,
			width: 25,
			height: 0,
			marker_size: 15
		},
		{
			name: 'Couche de sortie',
			neuron_count: 2,
			width: 20,
			height: 0,
			marker_size: 17,
			classes: Array.from({ length: 2 }, (_, i) => i.toString())
		}
	);
}
