import * as tf from '@tensorflow/tfjs';

export type DataBatch = {
	xs: tf.Tensor2D;
	labels: tf.Tensor2D;
};

export type DataSourceConfig = {
	imagesSpriteURL: string;
	labelsURL: string;
	imageSize: number;
	numClasses: number;
	numDatasetElements: number;
	numTrainElements: number;
};

export interface DataSource {
	nextTrainBatch: (batchSize: number) => DataBatch;
	nextTestBatch: (batchSize: number) => DataBatch;
	loadData: () => Promise<void>;
}

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 */
export class MnistData implements DataSource {
	private config: DataSourceConfig;
	private shuffledTrainIndex: number;
	private shuffledTestIndex: number;
	private datasetImages: Float32Array;
	private datasetLabels: Uint8Array;
	private trainIndices: Uint32Array;
	private testIndices: Uint32Array;
	private trainImages: Float32Array;
	private testImages: Float32Array;
	private trainLabels: Uint8Array;
	private testLabels: Uint8Array;

	constructor(config: DataSourceConfig) {
		this.config = config;
		this.shuffledTrainIndex = 0;
		this.shuffledTestIndex = 0;
	}

	async loadData() {
		// Make a request for the MNIST sprited image.
		const img = new Image();
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		const imgRequest = new Promise<void>((resolve, reject) => {
			img.crossOrigin = '';
			img.onload = () => {
				img.width = img.naturalWidth;
				img.height = img.naturalHeight;

				const chunkSize = 5000;

				const chunkCount = Math.ceil(this.config.numTrainElements / chunkSize);

				const datasetBytesBuffer = new ArrayBuffer(
					chunkCount * chunkSize * this.config.imageSize * 4
				);

				canvas.width = img.width;
				canvas.height = chunkSize;

				for (let i = 0; i < chunkCount; i++) {
					// console.log('chunk:', i);
					const datasetBytesView = new Float32Array(
						datasetBytesBuffer,
						i * this.config.imageSize * chunkSize * 4,
						this.config.imageSize * chunkSize
					);
					ctx.drawImage(img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width, chunkSize);

					const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

					for (let j = 0; j < imageData.data.length / 4; j++) {
						// All channels hold an equal value since the image is grayscale, so
						// just read the red channel.
						datasetBytesView[j] = imageData.data[j * 4] / 255;
					}
				}
				this.datasetImages = new Float32Array(datasetBytesBuffer);

				resolve();
			};
			img.src = this.config.imagesSpriteURL;
		});

		const labelsRequest = fetch(this.config.labelsURL);
		// eslint-disable-next-line @typescript-eslint/no-unused-vars
		const [imgResponse, labelsResponse] = await Promise.all([imgRequest, labelsRequest]);

		this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

		// Create shuffled indices into the train/test set for when we select a
		// random dataset element for training / validation.
		const numTestElements = this.config.numDatasetElements - this.config.numTrainElements;
		this.trainIndices = tf.util.createShuffledIndices(this.config.numTrainElements);
		this.testIndices = tf.util.createShuffledIndices(numTestElements);

		// Slice the the images and labels into train and test sets.
		this.trainImages = this.datasetImages.slice(
			0,
			this.config.imageSize * this.config.numTrainElements
		);
		this.testImages = this.datasetImages.slice(this.config.imageSize * numTestElements);
		this.trainLabels = this.datasetLabels.slice(
			0,
			this.config.numClasses * this.config.numTrainElements
		);
		this.testLabels = this.datasetLabels.slice(this.config.numClasses * numTestElements);
	}

	nextTrainBatch(batchSize: number): DataBatch {
		return this.nextBatch(batchSize, [this.trainImages, this.trainLabels], () => {
			this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
			return this.trainIndices[this.shuffledTrainIndex];
		});
	}

	nextTestBatch(batchSize: number): DataBatch {
		return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
			this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
			return this.testIndices[this.shuffledTestIndex];
		});
	}

	nextBatch(batchSize: number, data: [Float32Array, Uint8Array], index: () => number) {
		const batchImagesArray = new Float32Array(batchSize * this.config.imageSize);
		const batchLabelsArray = new Uint8Array(batchSize * this.config.numClasses);

		for (let i = 0; i < batchSize; i++) {
			const idx = index();

			const image = data[0].slice(
				idx * this.config.imageSize,
				idx * this.config.imageSize + this.config.imageSize
			);
			batchImagesArray.set(image, i * this.config.imageSize);

			const label = data[1].slice(
				idx * this.config.numClasses,
				idx * this.config.numClasses + this.config.numClasses
			);
			batchLabelsArray.set(label, i * this.config.numClasses);
		}

		const xs = tf.tensor2d(batchImagesArray, [batchSize, this.config.imageSize]);
		const labels = tf.tensor2d(batchLabelsArray, [batchSize, this.config.numClasses]);

		return { xs, labels };
	}
}
