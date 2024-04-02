import { asyncWritable, writable } from '@square/svelte-store';
import { MnistData } from '$lib/data';
import { newAllDigitsNetwork, type NetworkUnderTraining } from '$lib/models';

// Would really have loved to:
// - Add an asynreadable MNIST data store
// - Make the network store derive from that
// One advantage would have been that reloading the network store would not
// reload the MNIST data.
// Alas, I was not able to make that design work.

export const networkStore = asyncWritable<never[], NetworkUnderTraining>(
	[],
	async () => {
		const mnistData = new MnistData();
		await mnistData.load();
		return newAllDigitsNetwork(mnistData);
	},
	undefined,
	{ reloadable: true }
);

export const learningRateStore = writable<number>(0);
