<script lang="ts">
	import { onMount } from 'svelte';
	import type { MnistData } from '$lib/data.js';
	import { mnistDataStore, modelStore } from '../../stores';
	import { Grid, Button, Loader, Title, Text, Divider, Stack, Space } from '@svelteuidev/core';
	import ConfusionMatrix from '$lib/components/ConfusionMatrix.svelte';
	import * as tf from '@tensorflow/tfjs';

	let data: MnistData;
	let isLoading = true;
	let labelsAndPredictions: [number[], number[]];

	onMount(async () => {
		mnistDataStore.load().then((value) => {
			isLoading = false;
			data = value;
		});
	});

	const classes = ['Zero', 'Un', 'Deux', 'Trois', 'Quatre', 'Cinq', 'Six', 'Sept', 'Huit', 'Neuf'];

	function showAccuracy() {
		const testDataSize = 1000;

		labelsAndPredictions = tf.tidy(() => {
			const testData = data.nextTestBatch(testDataSize);
			const testxs = testData.xs.reshape([testDataSize, 28 * 28]);

			const labels = testData.labels.argMax(-1);
			const preds = $modelStore.predict(testxs).argMax(-1);

			return [labels.arraySync() as number[], preds.arraySync() as number[]];
		});
	}
</script>

<Title order={1}>Evaluer la précision du réseau</Title>
<Divider />
{#if isLoading}
	<Loader size="xl" />
{:else}
	<Grid cols={4}>
		<Grid.Col span={1}>
			<Space h="md" />
			<Stack>
				<Text
					>A chaque appui sur le bouton, on donne au réseau 1000 nouvelles images de <b>test</b> qu'il
					n'a jamais vues.
				</Text>
				<Button id="show-accuracy" on:click={showAccuracy}>Evaluer la précision</Button>
			</Stack>
		</Grid.Col>

		<Grid.Col span={3}>
			<p />
			<ConfusionMatrix {classes} {labelsAndPredictions} />
		</Grid.Col>
	</Grid>
{/if}
