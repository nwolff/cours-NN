<script lang="ts">
	import DrawBox from '$lib/components/DrawBox.svelte';
	import DistributionChart from '$lib/components/DistributionChart.svelte';
	import NetworkGraph from '$lib/components/NetworkGraph.svelte';
	import { Link } from '$lib/NetworkShape';
	import * as tf from '@tensorflow/tfjs';
	import { networkStore } from '../../stores';
	import { onMount } from 'svelte';
	import * as tslog from 'tslog';
	import NetworkStats from '$lib/components/NetworkStats.svelte';

	const logger = new tslog.Logger({ name: 'predict' });

	$: networkShape = $networkStore?.shape;
	$: labels = networkShape?.outputLayer.labels;
	let prediction: number[];
	let activations: number[][];

	let drawbox;

	$: weights = $networkStore?.tfModel.weights;

	let isLoading = true;
	onMount(async () => {
		await networkStore.load();
		isLoading = false;
	});

	function handleDrawnImage(event: { detail: { image: ImageData } }) {
		const image = event.detail.image;
		activations = calculateActivations(image);
		prediction = activations[activations.length - 1];
		logger.debug('tf.memory() ', tf.memory());
	}

	function calculateActivations(image: ImageData): number[][] {
		return tf.tidy(() => {
			const pixels = tf.browser.fromPixels(image, 1);

			// From: https://github.com/tensorflow/tfjs-examples/blob/master/webcam-transfer-learning/index.js
			const processedImage = tf
				.reshape(pixels, [1, 28 * 28])
				.toFloat()
				.div(255);

			const activationsTensor = $networkStore.featureModel.predict(processedImage) as tf.Tensor[];
			return [processedImage, ...activationsTensor].map((x) =>
				tf.squeeze(x).arraySync()
			) as number[][];
		});
	}

	function keepTopLinks(links: Link[]): Link[] {
		const length = links.length;
		if (length <= 500) {
			return links;
		}
		const sortedLinks = [...links].sort(
			(l1: Link, l2: Link) => Math.abs(l2.weight) - Math.abs(l1.weight)
		);
		return sortedLinks.slice(0, Math.min(500, 0.1 * length));
	}

	function applyActivation(links: Link[]): Link[] {
		if (!links.find((link) => link.a.activation)) {
			// A small optimization
			return links;
		}
		return links.map(
			(link) => new Link(link.a, link.b, link.weight * (1 + 0.5 * link.a.activation))
		);
	}

	function linkFilter(links: Link[]) {
		const linksWithActivationApplied = applyActivation(links);
		return keepTopLinks(linksWithActivationApplied);
	}
</script>

{#if isLoading}
	<span class="loading loading-spinner loading-lg text-primary"></span>
{:else}
	<div class="grid grid-cols-7 gap-4">
		<div class="col-span-2">
			<h4 class="text-xl mb-2">Dessiner un chiffre</h4>
			<DrawBox bind:this={drawbox} on:imageData={handleDrawnImage} />
			<button class="btn btn-outline btn-primary mt-4" on:click={drawbox.clear}>Effacer</button>
			<h4 class="text-xl mt-12 mb-2">Prédiction</h4>
			<DistributionChart {labels} percentages={prediction} />
			<h4 class="text-xl mt-12 mb-2">Statistiques</h4>
			<NetworkStats stats={$networkStore.stats} />
		</div>
		<div class="col-span-5">
			<NetworkGraph {networkShape} {activations} {weights} {linkFilter} />
		</div>
	</div>
{/if}
