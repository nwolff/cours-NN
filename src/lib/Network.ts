import { DenseNetwork } from '$lib/NetworkShape';
import type { TestResult } from '$lib/NetworkTesting';
import type { DataSource } from '$lib/DataSource';
import * as tf from '@tensorflow/tfjs';

type TrainingRound = {
	samplesSeen: number;
	finalAccuracy?: number;
};

export type NetworkStats = {
	samplesSeen: number;
	trainingAccuracy?: number;
	test: TestResult;
};

export class Network {
	readonly tfModel: tf.Sequential;
	readonly featureModel: tf.LayersModel;
	readonly shape: DenseNetwork;
	readonly stats: NetworkStats;

	readonly dataSource: DataSource;

	private trainingHistory: TrainingRound[] = [];

	constructor(tfModel: tf.Sequential, networkShape: DenseNetwork, dataSource: DataSource) {
		this.tfModel = tfModel;
		this.featureModel = toFeatureModel(tfModel);
		this.shape = networkShape;
		this.dataSource = dataSource;
		this.stats = {
			samplesSeen: 0,
			test: { classes: [], labels: [], predictions: [], accuracy: 0 }
		};
	}

	nextTrainBatch(batchSize: number) {
		return this.dataSource.nextTrainBatch(batchSize);
	}

	nextTestBatch(batchSize: number) {
		return this.dataSource.nextTestBatch(batchSize);
	}

	trainingRoundDone(trainingRound: TrainingRound) {
		this.trainingHistory.push(trainingRound);
		this.stats.samplesSeen += trainingRound.samplesSeen;
		this.stats.trainingAccuracy = trainingRound.finalAccuracy;
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
