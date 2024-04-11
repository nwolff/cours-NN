import * as tf from '@tensorflow/tfjs';
import type { Network } from '$lib/Network';
import { zip2 } from '$lib/generic/utils';

export type TestResult = {
	classes: string[];
	labels: number[];
	predictions: number[];
	accuracy: number;
};

export function testNetwork(network: Network, testDataSize: number): TestResult {
	const [labels, predictions] = tf.tidy(() => {
		const testData = network.nextTestBatch(testDataSize);
		const testxs = testData.xs.reshape([testDataSize, -1]);

		const labels = testData.labels.argMax(-1);
		const preds = network.tfModel.predict(testxs).argMax(-1);

		return [labels.arraySync() as number[], preds.arraySync() as number[]];
	});
	return {
		classes: network.shape.classes,
		labels: labels,
		predictions: predictions,
		accuracy: computeAccuracy(labels, predictions)
	};
}

function computeAccuracy(labels: number[], predictions: number[]): number {
	let allPredictions = 0;
	let correctPredictions = 0;
	for (const [label, prediction] of zip2(labels, predictions)) {
		allPredictions += 1;
		if (label == prediction) {
			correctPredictions += 1;
		}
	}
	return correctPredictions / allPredictions;
}
