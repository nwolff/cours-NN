import { asyncWritable, writable } from '@square/svelte-store';
import { newAllDigitsNetwork, newZeroOneNetwork, type NetworkUnderTraining } from '$lib/models';

export const networkStore = asyncWritable<never[], NetworkUnderTraining>(
	[],
	async () => {
		//const network = newZeroOneNetwork();
		const network = newAllDigitsNetwork();
		await network.dataSource.loadData();
		return network;
	},
	undefined,
	{ reloadable: true }
);

export const learningRateStore = writable<number>(0);
