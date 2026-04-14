import { writable } from 'svelte/store';
import { newAllDigitsNetwork } from '$lib/networks/AllDigitsNetwork';
import { newZeroOneNetwork } from '$lib/networks/ZeroOneNetwork';
import { newFashionNetwork } from '$lib/networks/FashionNetwork';
import { newTemperatureControlNetwork } from '$lib/networks/TemperatureControlNetwork';
import type { Network } from '$lib/Network';

function asyncReloadable<T>(loader: () => Promise<T>) {
	const { subscribe, set } = writable<T>(undefined as unknown as T);
	const load = async () => {
		const value = await loader();
		set(value);
	};
	return { subscribe, load, reload: load };
}

function persistable<T>(initialValue: T) {
	const { subscribe, set, update } = writable(initialValue);
	let current = initialValue;
	subscribe((v) => (current = v));
	return {
		subscribe,
		set,
		update,
		load: () => Promise.resolve(current)
	};
}

export const temperatureControlNetworkStore = asyncReloadable(async () =>
	newTemperatureControlNetwork()
);

export const zeroOnenetworkStore = asyncReloadable(async () => {
	const network = newZeroOneNetwork();
	await network.dataSource.loadData();
	return network;
});

export const allDigitsNetworkStore = asyncReloadable(async () => {
	const network = newAllDigitsNetwork();
	await network.dataSource.loadData();
	return network;
});

export const fashionNetworkStore = asyncReloadable(async () => {
	const network = newFashionNetwork();
	await network.dataSource.loadData();
	return network;
});

export const learningRateStore = persistable(0.2);
