import { DenseNetwork } from '$lib/NetworkShape';
import { Network } from '$lib/Network';
import { MnistData, type DataSourceConfig } from '$lib/DataSource';
import * as tf from '@tensorflow/tfjs';
import { base } from '$app/paths';

export function newFashionNetwork(): Network {
	return new Network(
		newFashionTFModel(),
		newFashionNetworkShape(),
		new MnistData(zeroOneDataSourceConfig)
	);
}

const zeroOneDataSourceConfig: DataSourceConfig = {
	imagesSpriteURL: base + '/fashion_images.png',
	labelsURL: base + '/fashion_labels_uint8',
	imageSize: 28 * 28,
	numClasses: 10,
	numDatasetElements: 60_000,
	numTrainElements: 50_000
};

function newFashionTFModel(): tf.Sequential {
	const model = tf.sequential();
	model.add(tf.layers.dense({ inputShape: [28 * 28], units: 50, activation: 'relu' }));
	model.add(tf.layers.dense({ units: 50, activation: 'relu' }));
	model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

	// This can be changed later, while training the model
	const optimizer = new tf.SGDOptimizer(0.1);

	model.compile({
		optimizer: optimizer,
		loss: 'categoricalCrossentropy',
		metrics: ['accuracy']
	});
	return model;
}

function newFashionNetworkShape(): DenseNetwork {
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
			neuron_count: 50,
			width: 55,
			height: 0,
			marker_size: 10
		},
		{
			name: 'Couche cachée 2',
			neuron_count: 50,
			width: 55,
			height: 0,
			marker_size: 10
		},
		{
			name: 'Couche de sortie',
			neuron_count: 10,
			width: 40,
			height: 5,
			marker_size: 16,
			classes: [
				'T-shirt',
				'Pantalon',
				'Pull',
				'Robe',
				'Manteau',
				'Sandale',
				'Chemise',
				'Basket',
				'Sac',
				'Bottine'
			],
			rotateClassNames: true
		}
	);
}
