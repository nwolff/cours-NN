import { asyncReadable, writable } from '@square/svelte-store';
import { MnistData, loadData } from '$lib/data';
import { newAllDigitsNetwork, newTwoDigitsNetwork, type NetworkUnderTraining } from '$lib/models';

export const mnistDataStore = asyncReadable<MnistData>(null, async () => {
	return await loadData();
});

export const networkStore = writable<NetworkUnderTraining>(newAllDigitsNetwork());
// export const networkStore = writable<NetworkUnderTraining>(newTwoDigitsNetwork());

export const learningRateStore = writable<number>(0);
