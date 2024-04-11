import { asyncWritable, writable } from '@square/svelte-store';
import { newAllDigitsNetwork } from '$lib/networks/AllDigitsNetwork';
import { newZeroOneNetwork } from '$lib/networks/ZeroOneNetwork';
import type { Network } from '$lib/Network';

export const zeroOnenetworkStore = asyncWritable<never[], Network>(
	[],
	async () => {
		const network = newZeroOneNetwork();
		await network.dataSource.loadData();
		return network;
	},
	undefined,
	{ reloadable: true }
);

export const allDigitsNetworkStore = asyncWritable<never[], Network>(
	[],
	async () => {
		const network = newAllDigitsNetwork();
		await network.dataSource.loadData();
		return network;
	},
	undefined,
	{ reloadable: true }
);

export const learningRateStore = writable<number>(0);
